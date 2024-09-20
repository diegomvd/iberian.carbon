from pystac_client import Client
from odc.stac import load
import odc.geo
import xarray as xr
import rioxarray as rio

# Using 10 main bands from Sentinel 2, discarding bands for atmospheric correction B1, B9 and B10.
# bands_sentinel2_10m = ['red', 'green', 'blue', 'nir']
# bands_sentinel2_20m = ['swir16','swir22','rededge1','rededge2','rededge3','nir08']

bands = ['red', 'green', 'blue', 'nir', 'swir16', 'swir22', 'rededge1', 'rededge2', 'rededge3', 'nir08']

# We are avoiding Sentinel-1 because its calibration is technically complex and more error-prone in rough terrain
# making it likely to end-up with biased products for the Iberian peninsula. 
# bands_sentinel1 = ['vv','vh'] 

# Filter the item_collection removing cloud percentage larger than 50%
cloud_filter_50percent = {
    "op": "lte",
    "args": [{"property": "eo:cloud_cover"}, 50]
}


# Think about processing everything with identical resolution
def process_patch(bbox: list, client: Client, year: str):

    growing_season = year # write a growing season string going from april to october every year
    
    search = client.search(
        max_items = 10, # Remove this line after testing locally
        collections = ['sentinel-2-l2a'],
        bbox = bbox, # Think about using intersects to get all spain and then use geobox to load progressively
        datetime = growing_season,
        filter = cloud_filter_50percent
    )
    item_collection = search.item_collection()    
    item_collection.save_object(f'sentinel2_items_{bbox}_{year}.json') # Save a modification of bbox where you say the coordinate

    src_dataset = odc.stac.load(
        item_collection,
        bands= bands,
        crs="EPSG:25830", # Reproject to the CRS used for DL
        resolution=10 # Change this after testing
    )
    # Calculate the median composite
    target_dataset= src_dataset.median(dim='time',skipna=True)
    target_dataset.rio.to_raster("test_multiband.tif")

# # Process 10m resolution data
# src_dataset = odc.stac.load(
#     item_collection,
#     bands= bands_sentinel2_10m,
#     crs="EPSG:25830", # Reproject to the CRS used for DL
#     resolution=400 # Change this after testing
# )

# # Calculate the median composite
# target_dataset_10m = src_dataset.median(dim='time',skipna=True)

# # Process 20m resolution data
# src_dataset = odc.stac.load(
#     item_collection,
#     bands= bands_sentinel2_20m,
#     crs="EPSG:25830",
#     resolution=400 # change this after testing
# )

# target_dataset_20m = src_dataset.median(dim='time',skipna=True)

# target_dataset = xr.merge([target_dataset_10m,target_dataset_20m]) 
# target_dataset.rio.to_raster("test_multiband.tif")