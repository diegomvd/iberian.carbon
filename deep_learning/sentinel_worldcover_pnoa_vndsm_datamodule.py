from lightning import LightningDataModule
from pnoa_vegetation_nDSM import PNOAnDSMV
from kornia_intersection_dataset import KorniaIntersectionDataset
import kornia.augmentation as K
from sentinel_worldcover_composites import Sentinel1,Sentinel2NDVI,Sentinel2RGBNIR,Sentinel2SWIR, SentinelWorldCoverYearlyComposites
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.samplers import RandomBatchGeoSampler, RandomGeoSampler, GridGeoSampler
from torchgeo.datasets import IntersectionDataset, GeoDataset
from torchgeo.datasets.splits import random_grid_cell_assignment
from torchgeo.datasets.utils import BoundingBox
from rasterio.crs import CRS
from torch.utils.data import _utils
from typing import Callable, Dict, Optional, Tuple, Type, Union
from torchgeo.datamodules import GeoDataModule


NODATA = {
    'B04': 0,
    'B02': 0,
    'B03': 0,
    'B08': 0,
    'B12-p50': 255,
    'B11-p50': 255,
    'NDVI-p90': 255,
    'NDVI-p50': 255,
    'NDVI-p10': 255,
    'VV': 0,
    'VH': 0,
    'ratio': 0,
}

OFFSET = {
    'B04': 0,
    'B02': 0,
    'B03': 0,
    'B08': 0,
    'B12-p50': 0,
    'B11-p50': 0,
    'NDVI-p90': -1,
    'NDVI-p50': -1,
    'NDVI-p10': -1,
    'VV': -45,
    'VH': -45,
    'ratio': -45,
}

SCALE = {
    'B04': 0.0001,
    'B02': 0.0001,
    'B03': 0.0001,
    'B08': 0.0001,
    'B12-p50': 0.004,
    'B11-p50': 0.004,
    'NDVI-p90': 0.008,
    'NDVI-p50': 0.008,
    'NDVI-p10': 0.008,
    'VV': 0.001,
    'VH': 0.001,
    'ratio': 0.001,
}

MINS = {
    'B04': 0,
    'B02': 0,
    'B03': 0,
    'B08': 0,
    'B12-p50': 0,
    'B11-p50': 0,
    'NDVI-p90': 0,
    'NDVI-p50': 0,
    'NDVI-p10': 0,
    'VV': -45,
    'VH': -45,
    'ratio': -45,
}

MAXS = {
    'B04': 1,
    'B02': 1,
    'B03': 1,
    'B08': 1,
    'B12-p50': 1,
    'B11-p50': 1,
    'NDVI-p90': 1,
    'NDVI-p50': 1,
    'NDVI-p10': 1,
    'VV': 20.535,
    'VH': 20.535,
    'ratio': 20.535,
}

class PNOAVnDSMRemoveAbnormalHeight(K.IntensityAugmentationBase2D):
    """Remove height outliers by assigning a no-data value equal to -1.0 meters"""

    hmax = 60.0 # Conserative height threshold

    def __init__(self, hmax: float = 60.0, nan_value: float = -1.0 ) -> None:
        super().__init__(p=1)
        self.nan_value = nan_value
        self.flags = {"hmax" : torch.tensor(hmax).view(-1,1,1)}

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        input[(input > flags['hmax'].to(torch.device("mps")))] = self.nan_value
        return input   

class PNOAVnDSMInputNoHeightInArtificialSurfaces(K.IntensityAugmentationBase2D):
    """Artificial surfaces are associated with a value of -32767.0, input a value of 0.0 meters so the NN learns to identify absence of vegetation because of artificial surface"""

    def __init__(self, nan_value: float = -32767.0 ) -> None:
        super().__init__(p=1)
        self.nan_value = nan_value
        self.flags = {"nodata" : torch.tensor(nan_value).view(-1,1,1)}

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        input[(input - flags['nodata'].to(torch.device("mps"))) == 0] = 0.0
        return input         

class SentinelWorldCoverRescale(K.IntensityAugmentationBase2D):
    """Rescale raster values according to scale and offset parameters"""

    def __init__(self, nodata: Tensor, offset: Tensor, scale: Tensor) -> None:
        super().__init__(p=1)
        self.flags = {"nodata" : nodata.view(-1,1,1), "offset": offset.view(-1,1,1), "scale": scale.view(-1,1,1)}

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        # Let NaN values be as they are in the data
        #input[(input - flags['nodata'].to(torch.device("mps"))) == 0] = float('nan') 
        return input * flags['scale'].to(torch.device("mps")) + flags['offset'].to(torch.device("mps"))        

class SentinelWorldCoverMinMaxNormalize(K.IntensityAugmentationBase2D):
    """Normalize Sentinel 1 GAMMA channels."""

    def __init__(self, mins: Tensor, maxs: Tensor) -> None:
        super().__init__(p=1)
        self.flags = {"min": mins.view(-1,1,1), "max": maxs.view(-1,1,1)}

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        return (input - flags["min"].to(torch.device("mps"))) / (flags["max"].to(torch.device("mps")) - flags["min"].to(torch.device("mps")) + 1e-8)        

class SentinelWorldCoverPNOAVnDSMDataModule(GeoDataModule):

    seed = 4356578
    predict_patch_size = 1000
    nan_value = -1.0

    # Could benefit from a parameter Nan in target to make sure that nodata value is not hardcoded.
    def __init__(self, data_dir: str = "path/to/dir", patch_size: int = 256, batch_size: int = 128, length: int = 10000, num_workers: int = 0, seed: int = 42, predict_patch_size: int = 12000):

        #  This is used to build the actual dataset.    
        self.data_dir = data_dir

        # When data is not downloaded and having a custom setup function the dataclass argument plays absolutely no role on anything. Choosing GeoDataset because it is parameterless.
        super().__init__(
            GeoDataset, batch_size, patch_size, length, num_workers
        )

        self.seed = seed
        self.predict_patch_size = predict_patch_size
        self.collate_fn = self.collate_geo

        nodata = torch.tensor([NODATA[b] for b in SentinelWorldCoverYearlyComposites.all_bands])
        offset = torch.tensor([OFFSET[b] for b in SentinelWorldCoverYearlyComposites.all_bands])
        scale = torch.tensor([SCALE[b] for b in SentinelWorldCoverYearlyComposites.all_bands])
        mins = torch.tensor([MINS[b] for b in SentinelWorldCoverYearlyComposites.all_bands])
        maxs = torch.tensor([MAXS[b] for b in SentinelWorldCoverYearlyComposites.all_bands])

        self.train_aug = {
            'general' : K.AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5, keepdim = True),
                K.RandomVerticalFlip(p=0.5, keepdim = True),
                #  Don't know if these augmentations make actual sense in a pixelwise regression context...
                #K.RandomAffine(degrees=(0, 360), scale=(0.3,0.9), p=0.25, keepdim = True),
                #K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.25, keepdim = True),
                data_keys=None,
                keepdim = True,
                same_on_batch = False, 
                random_apply=2
            ),
            'image' : K.AugmentationSequential(SentinelWorldCoverRescale(nodata,offset,scale), SentinelWorldCoverMinMaxNormalize(mins,maxs),data_keys=None,keepdim=True),
            'mask' : K.AugmentationSequential(PNOAVnDSMRemoveAbnormalHeight(),PNOAVnDSMInputNoHeightInArtificialSurfaces(),data_keys=None, keepdim=True)
        }

        self.aug = {
            'image' : K.AugmentationSequential(SentinelWorldCoverRescale(nodata,offset,scale), SentinelWorldCoverMinMaxNormalize(mins,maxs),data_keys=None,keepdim=True),
            'mask' : K.AugmentationSequential(PNOAVnDSMRemoveAbnormalHeight(),PNOAVnDSMInputNoHeightInArtificialSurfaces(),data_keys=None, keepdim=True)
        }

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Apply batch augmentations to the batch after it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """
        if self.trainer:
            if self.trainer.training:
                split = "train"
            elif self.trainer.validating or self.trainer.sanity_checking:
                split = "val"
            elif self.trainer.testing:
                split = "test"
            elif self.trainer.predicting:
                split = "predict"

            # TODO: EDIT this part to make it compatible with our dataset
            aug = self._valid_attribute(f"{split}_aug", "aug")

            # Remove geo  information
            del  batch['crs']
            del  batch['bbox']

            # Assign nan to nodata values
            # Image rescaling and normalization
            batch['image'] = aug['image']({'image':batch['image']})['image']

            #  Need to find a less confusing solution than calling image the mask
            batch['mask'] = aug['mask']({'image':batch['mask']})['image']
            
            if 'general' in aug.keys():
                # Image augmentation
                batch = aug['general'](batch)

        return batch

    def setup(self, stage: str):

        rgbnir_dataset = Sentinel2RGBNIR(self.data_dir)
        swir_dataset = Sentinel2SWIR(self.data_dir)
        ndvi_dataset = Sentinel2NDVI(self.data_dir)
        vvvhratio_dataset = Sentinel1(self.data_dir)

        sentinel = SentinelWorldCoverYearlyComposites(rgbnir_dataset,swir_dataset,ndvi_dataset,vvvhratio_dataset)

        pnoa_dataset = PNOAnDSMV(self.data_dir)
        print(pnoa_dataset)
        print(pnoa_dataset.bounds)

        self.nan_value = -1.0

        if stage in ['predict']:
            # Build the prediction dataset gathering copernicus data for portugal and spain 2020-2021
            self.predict_dataset = KorniaIntersectionDataset(sentinel, pnoa_dataset)
        
            self.predict_sampler = GridGeoSampler(
                self.predict_dataset, 256, 256
            ) 
            
        else:

            # This will downsample the canopy height data from 2,5m to 10m resolution.
            sentinel_pnoa = KorniaIntersectionDataset(sentinel, pnoa_dataset)
                        
            # Perform identical  splits by fixing the seed at fit and test stages to ensure that we are not training on test set.             
            self.train_dataset, self.val_dataset, self.test_dataset = random_grid_cell_assignment(
                sentinel_pnoa, [0.8,0.1,0.1], grid_size = 6, generator=torch.Generator().manual_seed(self.seed)
            )

            if stage in ['fit']:
                self.train_batch_sampler = RandomBatchGeoSampler(
                    self.train_dataset, self.patch_size, self.batch_size, self.length
                )

            if stage in ['fit', 'validate']:
                self.val_sampler = GridGeoSampler(
                    self.val_dataset, self.patch_size, self.patch_size
                )

            if stage in ['test']:
                self.test_sampler = GridGeoSampler(
                    self.test_dataset, self.patch_size, self.patch_size
                )     

    @staticmethod        
    def collate_crs_fn(batch,*,collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
        return batch[0]

    @staticmethod
    def collate_bbox_fn(batch, *,collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None ):
        return batch[0]

    @staticmethod
    def collate_geo(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
        collate_map = {torch.Tensor : _utils.collate.collate_tensor_fn, CRS : SentinelWorldCoverPNOAVnDSMDataModule.collate_crs_fn, BoundingBox : SentinelWorldCoverPNOAVnDSMDataModule.collate_bbox_fn}
        return _utils.collate.collate(batch, collate_fn_map=collate_map)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, dict):
            # move all tensors in your custom data structure to the device
            batch['image'] = batch['image'].to(device)
            batch['mask'] = batch['mask'].float().to(device)
        else:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch    
