import torch
from lightning.pytorch.callbacks import BasePredictionWriter
import rasterio
from rasterio.transform import from_bounds
import os

class CanopyHeightRasterWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        
        for i,predicted_patch in enumerate(prediction):
            index = batch_indices[i]
            print(index)
            transform = from_bounds(index.minx,index.miny,index.maxx,index.maxy,predicted_patch[0].shape[0],predicted_patch[0].shape[1])

            if index.mint<1609455600.0:
                year = 2020
            else:
                year = 2021 

            with rasterio.open(
                os.path.join(self.output_dir, str(dataloader_idx), f"predicted_batch_{batch_idx}_patch_{i}_{year}.tif"),
                mode="w",
                driver="GTiff",
                height=predicted_patch[0].shape[0],
                width=predicted_patch[0].shape[1],
                count=1,
                dtype= 'float32',
                crs="epsg:25830",
                transform=transform,
                nodata=-1.0,
                compress='lzw'    
            ) as new_dataset:
                new_dataset.write(predicted_patch[0], 1)
                new_dataset.update_tags(DATE = year)

        # In function of how comes the information just use the tensor image to build a raster with the corresponding raster bounds.

        # torch.save(prediction, os.path.join(self.output_dir, dataloader_idx, f"{batch_idx}.pt"))

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))
        return 'BLABLA'
