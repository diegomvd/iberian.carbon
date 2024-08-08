"""Sentinel-1 and Sentinel-2 annual composites for 2020 and 2021."""

from typing import Any, Callable, Optional, Union

from rasterio.crs import CRS

import kornia.augmentation as K
from torchgeo.datasets import RasterDataset, IntersectionDataset
from torchgeo.datasets.utils import BoundingBox
from typing import Any, cast

from functools import reduce

class SentinelWorldCoverRescale(K.IntensityAugmentationBase2D):
    """Rescale raster values according to scale and offset parameters"""

    def __init__(self, nodata: int, offset: float, scale: float) -> None:
        super().__init__(p=1)
        self.flags = {"offset": offset, "scale": scale}

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        input[input == flags['nodata']] = float('nan')
        return input * flags['scale'] + flags['offset']

class Sentinel1MinMaxNormalize(K.IntensityAugmentationBase2D):
    """Normalize Sentinel 1 GAMMA channels."""

    def __init__(self) -> None:
        super().__init__(p=1)
        self.flags = {"min": -44.0, "max": 20.535}

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        return (input - flags["min"]) / (flags["max"] - flags["min"] + 1e-6)

class SentinelWorldCoverYearlyComposites(GeoDataset):
    
    dataset = None

    all_bands = ['B04','B03','B02','B08','B12-p50','B11-p50','NDVI-p90','NDVI-p50','NDVI-p10','VV','VH','ratio']

    def intersect_datasets(dataset1: SentinelComposite, dataset2: SentinelComposite):
        return IntersectionDataset(dataset1,dataset2)

    def __init__(
        self,
        dataset_rgbnir: SentinelComposite,
        dataset_swir: SentinelComposite,
        dataset_ndvi: SentinelComposite,
        dataset_vvvhratio: SentinelComposite,       
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ) -> None:
        """Initialize a new SentinelWorldCoverYearlyComposites dataset instance.

        Args:
            datasets: Sentinel WorldCover datasets to join
            transforms: a function/transform that takes an input sample
                and returns a transformed version
        """
        self.dataset = dataset_rgbnir & dataset_swir & dataset_ndvi & dataset_vvvhratio
        super().__init__(transforms)

    def __getitem__(self, query: BoundingBox) -> Dict[str,Any]:

        sample = self.dataset.__getitem__(query)

        if self.transforms is not None:    
            sample['image'] = self.transforms(sample['image'])

        return sample    

class SentinelComposite(RasterDataset):
    is_image = True

    filename_regex = r'ESA_WorldCover_10m_(?P<date>\d{4})'
    date_format = "%Y"

    transforms_post_init = None

    def __init__(
        self,
        paths: Union[str, Iterable[str]] = "data",
        bands: Optional[Sequence[str]] = None,
        nodata: int = None,
        offset: float = None,
        scale: float = None,
    ) -> None:

        if self.filename_glob == "ESA_WorldCover_*S1VVVHratio*":
            self.transforms_post_init = K.AugmentationSequential(
                SentinelWorldCoverRescale(nodata,offset,scale),
                Sentinel1MinMaxNormalize(),
                data_keys=['image']
            )
        else:    
            self.transforms_post_init = K.AugmentationSequential(
                SentinelWorldCoverRescale(nodata,offset,scale),
                data_keys=['image'],
            )    
        super().__init__(paths=paths,bands=bands)


    def __getitem__(self, query: BoundingBox) -> Dict[str,Any]:

        sample = super().__getitem__(query)

        if self.transforms_post_init is not None:    
            sample['image'] = self.transforms_post_init(sample['image'])

        return sample    

class Sentinel2RGBNIR(SentinelComposite):
    """Sentinel-2 RGBNIR annual composites for 2020 and 2021.

    """

    filename_glob = "ESA_WorldCover_*S2RGBNIR*"
    all_bands = ['B04','B03','B02','B08']

    nodata = 0
    offset = 0
    scale = 0.0001

    def __init__(
        self,
        paths: Union[str, Iterable[str]] = "data",
    ) -> None:
        super.__init__(paths,self.all_bands,self.nodata,self.offset,self.scale)

class Sentinel2SWIR(SentinelComposite):
    """Sentinel-2 SWIR annual composites for 2020 and 2021.

    """

    filename_glob = "ESA_WorldCover_*SWIR*"
    all_bands = ['B12-p50','B11-p50']  

    nodata = 255
    offset = 0
    scale = 0.004

    def __init__(
        self,
        paths: Union[str, Iterable[str]] = "data",
    ) -> None:
        super.__init__(paths,self.all_bands,self.nodata,self.offset,self.scale)


class Sentinel2NDVI(SentinelComposite):
    """Sentinel-2 NDVI annual composites for 2020 and 2021.

    """
    filename_glob = "ESA_WorldCover_*NDVI*"
    all_bands = ['NDVI-p90', 'NDVI-p50', 'NDVI-p10']

    nodata = 255
    offset = -1
    scale = 0.008

    def __init__(
        self,
        paths: Union[str, Iterable[str]] = "data",
    ) -> None:
        super.__init__(paths,self.all_bands,self.nodata,self.offset,self.scale)

class Sentinel1(SentinelComposite):
    """Sentinel-1 VV, VH, VV/VH annual composites processed with GAMMA software for 2020 and 2021.

    """

    filename_glob = "ESA_WorldCover_*S1VVVHratio*"
    all_bands = ['VV','VH','ratio']

    nodata = 0
    offset = -45
    scale = 0.001

    def __init__(
        self,
        paths: Union[str, Iterable[str]] = "data",
    ) -> None:
        super.__init__(paths,self.all_bands,self.nodata,self.offset,self.scale)










    


    

