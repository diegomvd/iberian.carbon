from sentinel_worldcover_pnoa_vndsm_datamodule import SentinelWorldCoverPNOAVnDSMDataModule

from nan_robust_regression import NanRobustPixelWiseRegressionTask
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

import torch

path = '/Users/diegobengochea/git/iberian.carbon/data/LightningDataModule_Data_UTM30/'
# path = '/Users/diegobengochea/git/iberian.carbon/data/dl_test_utm30'

dm = SentinelWorldCoverPNOAVnDSMDataModule(data_dir=path)

# All tasks in TorchGeo use AdamW optimizer and LR decay on plateau by default.  
unet_regression = NanRobustPixelWiseRegressionTask(
    model='unet',
    backbone='resnet18',
    weights=None,
    in_channels=12, # Inventing an extra one can help getting pre trained weights for sentinel2
    num_outputs=1, 
    loss = 'mse',
    nan_value=dm.nan_value,
    lr = 0.001,
    patience =10    
)

# Define a lightning trainer
accelerator = 'mps' if torch.backends.mps.is_available() else 'cpu'
checkpoint_dir = ''
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss', dirpath=checkpoint_dir, save_top_k=1, save_last=True
)
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=20) # which min_delta to use?
tb_logger = TensorBoardLogger(save_dir=checkpoint_dir, name='canopyheight_logs')
csv_logger = CSVLogger(save_dir=checkpoint_dir, name='canopyheight_logs')

trainer = Trainer(
    check_val_every_n_epoch=1,
    accelerator=accelerator,
    devices="auto",
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=50,
    logger=csv_logger,
    max_epochs=1000,
)

trainer.fit(unet_regression, dm)
