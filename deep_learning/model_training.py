import os
from pathlib import Path
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from typing import Optional, Literal

from s2_pnoa_vegetation_datamodule import S2PNOAVegetationDataModule
from canopy_height_regression import CanopyHeightRegressionTask
from prediction_writer import CanopyHeightRasterWriter

class ModelTrainingPipeline:
    def __init__(
        self,
        data_dir: str,
        checkpoint_dir: str = "canopy_height_checkpoints",
        predict_patch_size: int = 8000,
        learning_rate: float = 0.001,
        patience: int = 10,
        early_stopping_patience: int = 100,
        max_epochs: int = 10000,
        val_check_interval: int = 1,
        log_steps: int = 50
    ):
        """
        Initialize the training pipeline with configuration parameters.
        
        Args:
            data_dir: Directory containing the dataset
            checkpoint_dir: Directory for saving checkpoints and logs
            predict_patch_size: Patch size for predictions
            learning_rate: Initial learning rate
            patience: Patience for learning rate scheduler
            early_stopping_patience: Patience for early stopping
            max_epochs: Maximum number of training epochs
            val_check_interval: Validation check frequency (epochs)
            log_steps: Frequency of logging steps
        """
        self.data_dir = Path(data_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = {
            "predict_patch_size": predict_patch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "early_stopping_patience": early_stopping_patience,
            "max_epochs": max_epochs,
            "val_check_interval": val_check_interval,
            "log_steps": log_steps
        }
        
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._setup_datamodule()
        self._setup_model()
        self._setup_trainer()

    def _setup_datamodule(self):
        """Initialize the data module."""
        self.datamodule = S2PNOAVegetationDataModule(
            data_dir=str(self.data_dir),
            predict_patch_size=self.config["predict_patch_size"]
        )

    def _setup_model(self):
        """Initialize the model."""
        self.model = CanopyHeightRegressionTask(
            nan_value=self.datamodule.hparams['nan_target'],
            lr=self.config["learning_rate"],
            patience=self.config["patience"]
        )

    def _setup_trainer(self):
        """Initialize the PyTorch Lightning trainer with callbacks and loggers."""
        # Set up accelerator
        accelerator = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=self.checkpoint_dir,
            save_top_k=3,
            save_last=True
        )
        
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=self.config["early_stopping_patience"]
        )
        
        pred_writer = CanopyHeightRasterWriter(
            spain_shapefile='/Users/diegobengochea/git/iberian.carbon/data/SpainPolygon/gadm41_ESP_0.shp',
            output_dir="canopy_height_predictions",
            write_interval="batch"
        )
        
        # Logger
        csv_logger = CSVLogger(
            save_dir=self.checkpoint_dir,
            name='logs'
        )
        
        self.trainer = Trainer(
            check_val_every_n_epoch=self.config["val_check_interval"],
            accelerator=accelerator,
            devices="auto",
            callbacks=[checkpoint_callback, early_stopping_callback, pred_writer],
            log_every_n_steps=self.config["log_steps"],
            logger=csv_logger,
            max_epochs=self.config["max_epochs"],
            num_sanity_val_steps=1  # Add this to check validation setup
        )

    def train(self, checkpoint_path: Optional[str] = None):
        """
        Train the model with optional checkpoint loading.
        
        Args:
            checkpoint_path: Optional path to checkpoint for resuming training
        """
        self.trainer.fit(self.model, self.datamodule, ckpt_path=checkpoint_path)

    def test(self, checkpoint_path: str):
        """
        Test the model using a specific checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint for testing
        """
        test_metrics = self.trainer.test(
            self.model,
            datamodule=self.datamodule,
            ckpt_path=checkpoint_path
        )
        return test_metrics

    def predict(self, checkpoint_path: str):
        """
        Generate predictions using a specific checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint for prediction
        """
        predictions = self.trainer.predict(
            self.model,
            datamodule=self.datamodule,
            ckpt_path=checkpoint_path
        )
        return predictions

def main():
    # Configuration
    DATA_DIR = '/Users/diegobengochea/git/iberian.carbon/data/S2_PNOA_DATASET/'
    CHECKPOINT_DIR = '/Users/diegobengochea/git/iberian.carbon/deep_learning/canopy_height_checkpoints/'
    CHECKPOINT_PATH = "/Users/diegobengochea/git/iberian.carbon/deep_learning/canopy_height_checkpoints/epoch=99-step=47600.ckpt"
    
    # Initialize pipeline
    pipeline = ModelTrainingPipeline(
        data_dir=DATA_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        learning_rate=0.001,
        patience=10,
        early_stopping_patience=100,
        max_epochs=10000
    )
    
    # Choose operation mode
    mode: Literal['train', 'test', 'predict'] = 'train'
    
    try:
        if mode == 'train':
            pipeline.train(checkpoint_path=CHECKPOINT_PATH)
        elif mode == 'test':
            test_metrics = pipeline.test(checkpoint_path=CHECKPOINT_PATH)
            print(f"Test metrics: {test_metrics}")
        elif mode == 'predict':
            predictions = pipeline.predict(checkpoint_path=CHECKPOINT_PATH)
            print("Predictions generated successfully")
    except Exception as e:
        print(f"Error during {mode}: {str(e)}")

if __name__ == "__main__":
    main()





