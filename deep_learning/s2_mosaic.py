"""Sentinel-2 green-season composites for 2017 to 2024."""

from torchgeo.datasets import RasterDataset


class S2Mosaic(RasterDataset):
    is_image = True

    filename_regex = r'S2_summer_mosaic_(?P<date>\d{4})'
    date_format = "%Y"

    all_bands = ['red', 'green', 'blue', 'nir', 'swir16', 'swir22', 'rededge1', 'rededge2', 'rededge3', 'nir08']

    separate_files = False

    nan_value = -9999

