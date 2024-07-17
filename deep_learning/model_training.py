from sentinel_worldcover_composites import Sentinel2SWIR, Sentinel2NDVI, Sentinel1, Sentinel2RGBNIR
from pnoa_vegetation_nDSM import PNOAnDSMV

from torchgeo.samplers import RandomBatchGeoSampler
from torch.utils.data import DataLoader

import kornia.augmentation as K
from kornia.augmentation.container import AugmentationSequential

from torchgeo.trainers import PixelwiseRegressionTask
from torchgeo.samplers import RandomBatchGeoSampler

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

path_swir = ''
path_rgbnir = ''
path_ndvi = ''
path_vvvhratio = ''
path_pnoa = ''

transforms = AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomAffine(degrees=(0, 90), p=0.25),
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.25),
    K.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), p=0.25),
    K.augmentation.ColorJitter(0.15, 0.25, 0.25, 0.25), # how to use this ???
    data_keys=[''],
    random_apply=10
)

swir_dataset = Sentinel2SWIR(path_swir, transforms=transforms)
rgbnir_dataset = Sentinel2RGBNIR(path_rgbnir, transforms=transforms)
ndvi_dataset = Sentinel2NDVI(path_ndvi, transforms=transforms)
vvvhratio_dataset = Sentinel1(path_vvvhratio, transforms=transforms)
pnoa_dataset = PNOAnDSMV(path_pnoa) # shoudl i transfotm target also??

dataset = swir_dataset & rgbnir_dataset & ndvi_dataset & vvvhratio_dataset & pnoa_dataset

sampler = RandomBatchGeoSampler(dataset, size = 256*256, batch_size = 128, length = 10000)

dataloader = DataLoader(dataset, sampler=sampler, num_workers=0)


regression = PixelwiseRegressionTask(
    model='Unet', # does not support FCSiamDiff
    backbone='resnet18',
    weights=None,
    in_channels=10,
    num_outputs=1, 
    loss = 'mse',
    lr = 0.0001,
    patience =10    
)

# Define a lightning trainer
accelerator = 'mps' if torch.mps.is_available() else 'cpu'
checkpoint_dir = ''
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss', dirpath=save_dir, save_top_k=1, save_last=True
)
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10) # which min_delta to use?
tb_logger = TensorBoardLogger(save_dir=checkpoint_dir, name='canopyheight_logs')
csv_logger = CSVLogger(save_dir=checkpoint_dir, name='canopyheight_logs')

trainer = Trainer(
    accelerator=accelerator,
    callbacks=[checkpoint_callback, early_stopping_callback],
    fast_dev_run=fast_dev_run,
    log_every_n_steps=1,
    logger=tb_logger,
    min_epochs=1,
    max_epochs=max_epochs,
)

trainer.fit(model=regression, train_dataloaders=dataloader)






