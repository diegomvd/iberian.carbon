"""Sentinel-2 green-season composites for 2017 to 2024."""

from torchgeo.datasets import RasterDataset

class Sentinel2Composite(RasterDataset):
    is_image = True

    filename_glob = "sentinel2_mosaic_*"
    filename_regex = r'sentinel2_mosaic_(?P<date>\d{4})'
    date_format = "%Y"

    all_bands = ['red','green','blue','nir','swir16','swir22','rededge1','rededge2','rededge3','nir08']

    separate_files = False

    nan_value = 0
