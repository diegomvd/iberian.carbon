import os
from pathlib import Path
import torch
from lightning.pytorch import Trainer
from s2_pnoa_vegetation_datamodule import S2PNOAVegetationDataModule
from canopy_height_regression import CanopyHeightRegressionTask
from prediction_writer import CanopyHeightRasterWriter

# Configuration
DATA_DIR = '/Users/diegobengochea/git/iberian.carbon/data/S2_PNOA_DATASET/'
CHECKPOINT_PATH = '/path/to/your/checkpoint.ckpt'  # Replace with actual path
OUTPUT_DIR = 'canopy_height_predictions'
PREDICTION_PATCH_SIZE = 6144
STRIDE = 256

class PredictionPipeline:
    def __init__(
        self,
        data_dir: str,
        checkpoint_path: str,
        output_dir: str,
        patch_size: int = 6144,
        stride: int = 256  # Overlap between predictions
    ):
        self.data_dir = Path(data_dir)
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.patch_size = patch_size
        self.stride = stride
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._setup_datamodule()
        self._setup_model()
        self._setup_trainer()
        
    def _setup_datamodule(self):
        """Initialize prediction-specific datamodule configuration."""
        self.datamodule = S2PNOAVegetationDataModule(
            data_dir=str(self.data_dir),
            predict_patch_size=self.patch_size
        )
        
    def _setup_model(self):
        """Load the model from checkpoint."""
        self.model = CanopyHeightRegressionTask.load_from_checkpoint(
            self.checkpoint_path,
            nan_value_target=self.datamodule.hparams['nan_target'],
            nan_value_input=self.datamodule.hparams['nan_input']
        )
        self.model.eval()  # Set to evaluation mode
        
    def _setup_trainer(self):
        """Configure trainer specifically for prediction."""
        pred_writer = CanopyHeightRasterWriter(
            output_dir=str(self.output_dir),
            write_interval="batch"
        )
        
        self.trainer = Trainer(
            accelerator='mps' if torch.backends.mps.is_available() else 'cpu',
            devices="auto",
            callbacks=[pred_writer],
            enable_progress_bar=True,
            inference_mode=True,  # Important for memory efficiency
            logger=False  # No need for logging during prediction
        )
        
    def predict(self):
        """Run predictions."""
        try:
            print(f"Starting predictions with patch size: {self.patch_size}")
            print(f"Loading checkpoint: {self.checkpoint_path}")
            print(f"Output directory: {self.output_dir}")
            
            predictions = self.trainer.predict(
                self.model,
                datamodule=self.datamodule
            )
            
            print("Predictions completed successfully")
            return predictions
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise e

def main():
    
    # Initialize and run prediction
    pipeline = PredictionPipeline(
        data_dir=DATA_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        output_dir=OUTPUT_DIR,
        patch_size=PREDICTION_PATCH_SIZE,  # Adjust based on your memory constraints
        stride=STRIDE        # Adjust overlap as needed
    )
    
    pipeline.predict()

if __name__ == "__main__":
    main()