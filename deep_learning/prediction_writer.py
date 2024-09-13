import torch
from lightning.pytorch.callbacks import BasePredictionWriter

class CanopyHeightRasterWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        
        print('BATCH INDEX')
        print(batch_idx)
        print('BATCH INDICES')
        print(batch.keys())
        print(batch['image'].shape)
        print(len(batch_indices))
        print(batch_indices[0])
        print('DATALOADER INDEX')
        print(dataloader_idx)

        # In function of how comes the information just use the tensor image to build a raster with the corresponding raster bounds.

        # torch.save(prediction, os.path.join(self.output_dir, dataloader_idx, f"{batch_idx}.pt"))

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))
        return 'BLABLA'
