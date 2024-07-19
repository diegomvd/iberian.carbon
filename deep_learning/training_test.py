from sentinel_worldcover_composites import Sentinel2SWIR, Sentinel2NDVI, Sentinel1, Sentinel2RGBNIR
from pnoa_vegetation_nDSM import PNOAnDSMV

from torch.utils.data import DataLoader, random_split

import kornia.augmentation as K
from kornia.augmentation.container import AugmentationSequential

from torchgeo.samplers import RandomBatchGeoSampler
from torchgeo.trainers import PixelwiseRegressionTask
from torchgeo.samplers import RandomBatchGeoSampler, RandomGeoSampler
from torchgeo.datasets import IntersectionDataset

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

import torch

print('Starting')

path_sentinel = '/Users/diegobengochea/git/iberian.carbon/data/WorldCover_composites_2020_2021_test/'
path_pnoa = '/Users/diegobengochea/git/iberian.carbon/data/Vegetation_NDSM_PNOA2/PNOA2_merged_UTM30_test/'

print('Declaring augmentation list')
aug_list = AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomAffine(degrees=(0, 360), scale=(0.3,0.9), p=0.25),
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.25),
    K.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0), p=0.25),
    # data_keys=None,
    data_keys = ['image','mask'],
    random_apply=3
)

print('Loading image data')
# Image data
swir_dataset = Sentinel2SWIR(path_sentinel)
rgbnir_dataset = Sentinel2RGBNIR(path_sentinel)
ndvi_dataset = Sentinel2NDVI(path_sentinel)
vvvhratio_dataset = Sentinel1(path_sentinel)
print('Loading mask data')
# Mask data
pnoa_dataset = PNOAnDSMV(path_pnoa)

print('Intersecting datasets')
# SWIR dataset is put at the end to upsample it from 20m to 10m resolution instead of downsampling the rest
dataset_image = rgbnir_dataset & ndvi_dataset & vvvhratio_dataset & swir_dataset
# This will downsample the canopy height data from 2,5m to 10m resolution.
dataset = IntersectionDataset(dataset_image, pnoa_dataset)

print('Dataset')
print(dataset)

print('Defining sampler')
sampler = RandomGeoSampler(dataset, size=256, length=2)

# print('Performing splits in train, validation and test sets')
train_set, val_set = random_split(dataset=dataset,lengths=[0.7,0.3])

print('Instantiate dataloaders')
train_dataloader = DataLoader(train_set, sampler=sampler, num_workers=0)
val_dataloader = DataLoader(val_set, sampler=sampler, num_workers=0)
# test_dataloader = DataLoader(test_set, sampler=sampler, num_workers=0)

print('Declaring the model')
# All tasks in TorchGeo use AdamW optimizer and LR decay on plateau by default.  
unet_regression = PixelwiseRegressionTask(
    model='unet',
    backbone='resnet18',
    weights=None,
    in_channels=12, # Inventing an extra one can help getting pre trained weights for sentinel2
    num_outputs=1, 
    loss = 'mse',
    lr = 0.001,
    patience =10    
)

print('Defining lightning trainer')
# Define a lightning trainer
accelerator = 'mps' if torch.backends.mps.is_available() else 'cpu'
checkpoint_dir = ''
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss', dirpath=checkpoint_dir, save_top_k=1, save_last=True
)
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10) # which min_delta to use?
tb_logger = TensorBoardLogger(save_dir=checkpoint_dir, name='canopyheight_logs')
csv_logger = CSVLogger(save_dir=checkpoint_dir, name='canopyheight_logs')


trainer = Trainer(
    accelerator=accelerator,
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=50,
    logger=tb_logger,
    max_epochs=2,
)

print('Starting training process')
#trainer.fit(model=unet_regression, train_dataloaders=train_dataloader)

trainer.fit(model=unet_regression, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# print('Starting model testing')
# trainer.test(model=unet_regression, dataloaders=test_dataloader)
