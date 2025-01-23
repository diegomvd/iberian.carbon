import torch
from lightning.pytorch.callbacks import BasePredictionWriter
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import geopandas as gpd
from shapely import geometry
import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np

class CanopyHeightRasterWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self._setup_spain_mask()
        
    def _setup_spain_mask(self):
        """Load Spain shapefile once during initialization"""
        self.spain = gpd.read_file(
            '/Users/diegobengochea/git/iberian.carbon/data/SpainPolygon/gadm41_ESP_0.shp'
        )
        self.spain['geometry'] = self.spain['geometry'].to_crs('epsg:25830')

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        # Process single prediction (batch_size=1)
        predicted_patch = prediction[0]
        index = batch_indices[0]
        
        try:
            self._process_single_prediction(
                predicted_patch, 
                index,
                dataloader_idx,
                batch_idx
            )
        finally:
            # For MPS, explicit deletion is usually enough due to unified memory
            if isinstance(predicted_patch, torch.Tensor):
                del predicted_patch
                # No need for cuda.empty_cache() with MPS
                
                # Optional: Force garbage collection if memory pressure is high
                # import gc
                # gc.collect()

    def _process_single_prediction(self, predicted_patch, index, dataloader_idx, batch_idx):
        # Check Spain intersection
        tile = geometry.box(
            minx=index.minx, 
            maxx=index.maxx, 
            miny=index.miny, 
            maxy=index.maxy
        )

        if not intersects(self.spain['geometry'], tile).any():
            return

        # Setup output path
        savepath = Path(os.path.join(
            self.output_dir, 
            str(dataloader_idx), 
            f"predicted_minx_{index.minx}_maxy_{index.maxy}_mint_{index.mint}.tif"
        ))
        savepath.parent.mkdir(parents=True, exist_ok=True)

        # Move to CPU and convert to numpy
        with torch.no_grad():
            if isinstance(predicted_patch, torch.Tensor):
                # For MPS, we just need to move to CPU
                pred_data = predicted_patch[0].cpu().numpy()
            else:
                pred_data = predicted_patch[0][0].cpu().numpy()

        # Calculate transform
        transform = from_bounds(
            index.minx, 
            index.miny, 
            index.maxx, 
            index.maxy, 
            pred_data.shape[0], 
            pred_data.shape[1]
        )

        # Write to disk with optimal tile size for large patches
        with rasterio.open(
            savepath,
            mode="w",
            driver="GTiff",
            height=pred_data.shape[0],
            width=pred_data.shape[1],
            count=1,
            dtype='float32',
            crs="epsg:25830",
            transform=transform,
            nodata=-1.0,
            compress='lzw',
            tiled=True,
            blockxsize=256,  # Optimized for common tile sizes
            blockysize=256    
        ) as new_dataset:
            new_dataset.write(pred_data, 1)
            new_dataset.update_tags(DATE=index.mint)

        # Clear data
        del pred_data
        
    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        return None