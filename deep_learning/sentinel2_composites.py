"""Sentinel-2 green-season composites for 2017 to 2024."""

from torchgeo.datasets import RasterDataset, IntersectionDataset


    

        

class Sentinel2Composite(RasterDataset):
    is_image = True

    filename_regex = r'pnt_sentinel2_(?P<date>\d{4})'
    date_format = "%Y"

class Sentinel2RGB(Sentinel2Composite):

    filename_glob = "pnt_sentinel2_*_b432_*"

    all_bands = ['Red','Green','Blue']

    separate_files = False

    #nan_value = 0

class Sentinel2IRC(Sentinel2Composite):

    filename_glob = "pnt_sentinel2_*_b843_*"

    all_bands = ['Nir']

    separate_files = False

    #nan_value = 0    

class Sentinel2(IntersectionDataset):
    all_bands = ['Red','Green','Blue','Nir']

    def __init__(

        self,
        dataset_rgb: Sentinel2RGB,
        dataset_irc: Sentinel2IRC,
    ) -> None:

        super().__init__(dataset_rgb,dataset_irc)    

