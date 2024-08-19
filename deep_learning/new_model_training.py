from sentinel_wordlcover_pnoa_vndsm_datamodule import SentinelWorldCoverPNOAVnDSMDataModule

from torchgeo.trainers import PixelwiseRegressionTask

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

import torch

path = '/Users/diegobengochea/git/iberian.carbon/data/LightningDataModule_Data/'
path = '/Users/diegobengochea/git/iberian.carbon/data/dl_test_utm30'

dm = SentinelWorldCoverPNOAVnDSMDataModule(data_dir=path)

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

print(torch.mps.device_count())

trainer = Trainer(
    accelerator=accelerator,
    devices='0',
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=50,
    logger=tb_logger,
    max_epochs=1000,
)

print('Starting training process')

trainer.fit(unet_regression, dm)
