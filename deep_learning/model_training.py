from sentinel_worldcover_composites import Sentinel2SWIR, Sentinel2NDVI, Sentinel1, Sentinel2RGBNIR
from pnoa_vegetation_nDSM import PNOAnDSMV

from torch.utils.data import DataLoader, random_split

import kornia.augmentation as K
from kornia.augmentation.container import AugmentationSequential

from torchgeo.samplers import RandomBatchGeoSampler
from torchgeo.trainers import PixelwiseRegressionTask
from torchgeo.samplers import RandomBatchGeoSampler
from torchgeo.datasets import IntersectionDataset

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger


path_swir = ''
path_rgbnir = ''
path_ndvi = ''
path_vvvhratio = ''
path_pnoa = ''

aug_list = AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomAffine(degrees=(0, 360), scale=(0.3,0.9), p=0.25),
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.25),
    K.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0), p=0.25),
    data_keys=None,
    random_apply=3
)

# Image data
swir_dataset = Sentinel2SWIR(path_swir)
rgbnir_dataset = Sentinel2RGBNIR(path_rgbnir)
ndvi_dataset = Sentinel2NDVI(path_ndvi)
vvvhratio_dataset = Sentinel1(path_vvvhratio)

# Mask data
pnoa_dataset = PNOAnDSMV(path_pnoa)

dataset_image = rgbnir_dataset & ndvi_dataset & vvvhratio_dataset & swir_dataset
dataset = IntersectionDataset(dataset_image, pnoa_dataset, transforms=aug_list)

sampler = RandomBatchGeoSampler(dataset, size = 256, batch_size = 128, length = 10000)

train_set, val_set, test_set = random_split(dataset=dataset,lengths=[0.8,0.1,0.1])

train_dataloader = DataLoader(train_set, sampler=sampler, num_workers=0)
val_dataloader = DataLoader(val_set, sampler=sampler, num_workers=0)
test_dataloader = DataLoader(test_set, sampler=sampler, num_workers=0)

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

# Define a lightning trainer
accelerator = 'mps' if torch.mps.is_available() else 'cpu'
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
    max_epochs=1000,
)

trainer.fit(model=unet_regression, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer.test(model=unet_regression, dataloaders=test_dataloader)






