"""Sentinel-1 and Sentinel-2 annual composites for 2020 and 2021."""

from typing import Any, Callable, Optional, Union

from rasterio.crs import CRS

import kornia.augmentation as K
from torchgeo.datasets import RasterDataset, IntersectionDataset
from torchgeo.datasets.utils import BoundingBox
from typing import Any, cast

class SentinelWorldCoverYearlyComposites(GeoDataset):
    
    dataset = None

    all_bands = ['B04','B03','B02','B08','B12-p50','B11-p50','NDVI-p90','NDVI-p50','NDVI-p10','VV','VH','ratio']

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

    def __init__(
        self,
        paths: Union[str, Iterable[str]] = "data",
        bands: Optional[Sequence[str]] = None,
    ) -> None:

        super().__init__(paths=paths,bands=bands)

class Sentinel2RGBNIR(SentinelComposite):
    """Sentinel-2 RGBNIR annual composites for 2020 and 2021.

    """

    filename_glob = "ESA_WorldCover_*S2RGBNIR*"
    all_bands = ['B04','B03','B02','B08']

    def __init__(
        self,
        paths: Union[str, Iterable[str]] = "data",
    ) -> None:
        super.__init__(paths,self.all_bands)

class Sentinel2SWIR(SentinelComposite):
    """Sentinel-2 SWIR annual composites for 2020 and 2021.

    """

    filename_glob = "ESA_WorldCover_*SWIR*"
    all_bands = ['B12-p50','B11-p50']  

    def __init__(
        self,
        paths: Union[str, Iterable[str]] = "data",
    ) -> None:
        super.__init__(paths,self.all_bands)


class Sentinel2NDVI(SentinelComposite):
    """Sentinel-2 NDVI annual composites for 2020 and 2021.

    """
    filename_glob = "ESA_WorldCover_*NDVI*"
    all_bands = ['NDVI-p90', 'NDVI-p50', 'NDVI-p10']

    def __init__(
        self,
        paths: Union[str, Iterable[str]] = "data",
    ) -> None:
        super.__init__(paths,self.all_bands)

class Sentinel1(SentinelComposite):
    """Sentinel-1 VV, VH, VV/VH annual composites processed with GAMMA software for 2020 and 2021.

    """

    filename_glob = "ESA_WorldCover_*S1VVVHratio*"
    all_bands = ['VV','VH','ratio']

    def __init__(
        self,
        paths: Union[str, Iterable[str]] = "data",
    ) -> None:
        super.__init__(paths,self.all_bands)










    


    

