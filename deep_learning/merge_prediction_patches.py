import rasterio 
import numpy as np 
from rasterio.merge import merge 
from pathlib import Path


path = "/Users/diegobengochea/git/iberian.carbon/deep_learning/predictions_canopy_height/0/"

# raster_files = [file for file in Path(path).glob('*.tif') if year in str(file)]

for year in ['2020', '2021']:


    raster_files = [file for file in Path(path).glob(f'*_batch_0_*{year}.tif') ] 

    print(raster_files)

    with rasterio.open(raster_files[0]) as src:
        print(src.bounds)
        print(src.height)

    image_sum, transform_sum = merge(raster_files, method = 'sum')

    print(image_sum.shape)

    image_count, transform_count = merge(raster_files, method = 'count')

    image = image_sum/image_count

    print(image.shape)

    savepath = "/Users/diegobengochea/git/iberian.carbon/deep_learning/predictions_canopy_height/merged/canopy_height_{}.tif".format(year)
    with rasterio.open(
            savepath,
            mode="w",
            driver="GTiff",
            height=image.shape[-2],
            width=image.shape[-1],
            count=1,
            dtype= 'float32',
            crs="epsg:25830",
            transform=transform_count,
            nodata=-1.0,
            compress='lzw'    
        ) as new_dataset:
            new_dataset.write(image[0,0,:,:], 1)
            new_dataset.update_tags(DATE = year)

