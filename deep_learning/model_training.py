from sentinel2_pnoa_vndsm_datamodule import Sentinel2PNOAVnDSMDataModule

from nan_robust_regression import NanRobustPixelWiseRegressionTask, NanRobustHeightThresholdPixelWiseRegressionTask
from prediction_writer import CanopyHeightRasterWriter

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

import torch

# import ray.train.lightning
# from ray.train.torch import TorchTrainer

path = '/Users/diegobengochea/git/iberian.carbon/data/training_data_Sentinel2_PNOA_UTM30/'
# path = '/Users/diegobengochea/git/iberian.carbon/data/dl_test_utm30'

dm = Sentinel2PNOAVnDSMDataModule(data_dir=path)

# All tasks in TorchGeo use AdamW optimizer and LR decay on plateau by default.  
unet_regression = NanRobustHeightThresholdPixelWiseRegressionTask(
    model='unet',
    backbone='resnet18',
    weights=None,
    in_channels=10, 
    num_outputs=1, 
    loss = 'mse',
    nan_value=dm.nan_value,
    lr = 0.001,
    patience = 10    
)

# Define a lightning trainer
accelerator = 'mps' if torch.backends.mps.is_available() else 'cpu'
checkpoint_dir = ''
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss', dirpath=checkpoint_dir, save_top_k=1, save_last=True
)
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=50) # which min_delta to use?
# tb_logger = TensorBoardLogger(save_dir=checkpoint_dir, name='canopyheight_logs')
csv_logger = CSVLogger(save_dir=checkpoint_dir, name='canopyheight_logs')

pred_writer = CanopyHeightRasterWriter(output_dir="predictions_canopy_height", write_interval="batch")

trainer = Trainer(
    check_val_every_n_epoch=1,
    accelerator=accelerator,
    # accelerator = 'auto',
    devices="auto",
    callbacks=[checkpoint_callback, early_stopping_callback, pred_writer],
    log_every_n_steps=50,
    logger=csv_logger,
    max_epochs=10000,
)

resume_from_checkpoint = False
stage = 'predict' 
#stage = 'test'
#stage = 'fit'


if resume_from_checkpoint:
    if stage=='test':
        test_metrics = trainer.test(unet_regression, datamodule = dm, ckpt_path ="/Users/diegobengochea/git/iberian.carbon/deep_learning/epoch=2-step=234.ckpt")
        print(test_metrics)
    elif stage == 'predict':
        prediction = trainer.predict(unet_regression, datamodule = dm, ckpt_path = "/Users/diegobengochea/git/iberian.carbon/deep_learning/model_0_weights/epoch=2-step=234.ckpt")
        print(prediction)
    else:
        trainer.fit(unet_regression, datamodule=dm, ckpt_path="/Users/diegobengochea/git/iberian.carbon/deep_learning/model_0_weights/epoch=2-step=234.ckpt")
else:
    trainer.fit(unet_regression, dm)



