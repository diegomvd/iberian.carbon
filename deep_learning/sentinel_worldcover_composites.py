"""Sentinel-1 and Sentinel-2 annual composites for 2020 and 2021."""

from typing import Any, Callable, Optional, Union

from rasterio.crs import CRS

import kornia.augmentation as K
from torchgeo.datasets import RasterDataset, IntersectionDataset, GeoDataset
from torchgeo.datasets.utils import BoundingBox
from typing import Any, cast

class SentinelComposite(RasterDataset):
    is_image = True

    filename_regex = r'ESA_WorldCover_10m_(?P<date>\d{4})'
    date_format = "%Y"

class SentinelWorldCoverYearlyComposites(IntersectionDataset):
    
    all_bands = ['B04','B03','B02','B08','B12-p50','B11-p50','NDVI-p90','NDVI-p50','NDVI-p10','VV','VH','ratio']

    def __init__(
        self,
        dataset_rgbnir: SentinelComposite,
        dataset_swir: SentinelComposite,
        dataset_ndvi: SentinelComposite,
        dataset_vvvhratio: SentinelComposite,
    ) -> None:
        """Initialize a new SentinelWorldCoverYearlyComposites dataset instance.

        Args:
            datasets: Sentinel WorldCover datasets to join
            transforms: a function/transform that takes an input sample
                and returns a transformed version
        """
        # First create an intersection dataset of the first 3 datasets
        dataset = dataset_rgbnir & dataset_swir & dataset_ndvi
 
        super().__init__(dataset,dataset_vvvhratio)    


class Sentinel2RGBNIR(SentinelComposite):
    """Sentinel-2 RGBNIR annual composites for 2020 and 2021.

    """
    filename_glob = "ESA_WorldCover_*S2RGBNIR*"
    all_bands = ['B04','B03','B02','B08']

class Sentinel2SWIR(SentinelComposite):
    """Sentinel-2 SWIR annual composites for 2020 and 2021.

    """
    filename_glob = "ESA_WorldCover_*SWIR*"
    all_bands = ['B12-p50','B11-p50']  


class Sentinel2NDVI(SentinelComposite):
    """Sentinel-2 NDVI annual composites for 2020 and 2021.

    """
    filename_glob = "ESA_WorldCover_*NDVI*"
    all_bands = ['NDVI-p90', 'NDVI-p50', 'NDVI-p10']


class Sentinel1(SentinelComposite):
    """Sentinel-1 VV, VH, VV/VH annual composites processed with GAMMA software for 2020 and 2021.

    """
    filename_glob = "ESA_WorldCover_*S1VVVHratio*"
    all_bands = ['VV','VH','ratio']











    


    

