
from odc.stac import load
import odc.geo
import xarray as xr
import rioxarray as rio
import geopandas as gpd 
import itertools
import dask.distributed
import dask.utils
from odc.stac import configure_rio, stac_load
#from odc.algo import erase_bad, mask_cleanup
import os
import threading
from pystac_client import Client
from pathlib import Path
import numpy as np
import logging
import gc
import os
import psutil
import time
from contextlib import contextmanager

N_SCENES = 12
CHUNK_SIZE = 2048
CLOUD_THRESHOLD = 1
YEARS = [2017,2018,2019,2020,2021,2022,2023,2024]
MIN_MONTH = 6
MAX_MONTH = 9
STAC_URL = "https://earth-search.aws.element84.com/v1"
TILE_SIZE = 2048*6
BANDS = ['red', 'green', 'blue', 'nir', 'swir16', 'swir22', 'rededge1', 'rededge2', 'rededge3', 'nir08','scl']
BANDS_DROP = ['scl']

MIN_N_ITEMS = 40
MAX_COUD_THRESHOLD = 61

N_WORKERS=8
THREADS_PER_WORKER=3
MEM_PER_WORKER='20Gb'


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_mosaic(dataset,year,sampling_period):

    print(len(dataset.time))
    dataset, time_span = select_best_scenes(dataset,N_SCENES)
    
    # Apply correction factor for scenes using proccessing baseline 04.00
    if year>2021:
        dataset = dataset - 1000
    
    mosaic = dataset.median(dim="time",skipna=True)

    mosaic.attrs['valid_pixel_percentage'] = mosaic.count()/(len(mosaic.x)*len(mosaic.y))*100
    mosaic.attrs['time_span'] = time_span
    mosaic.attrs['year'] = year
    mosaic.attrs['sampling_period'] = sampling_period
    
    mosaic = mosaic.fillna(0).astype('uint16')

    for band in mosaic.variables:
        mosaic[band] = mosaic[band].rio.write_nodata(0, inplace=False)

        
    return mosaic    

@contextmanager
def setup_optimized_cluster():
    """
    Setup Dask cluster optimized for 24 cores and 192GB RAM
    """
    
    cluster = None
    client = None
    try:
        cluster = dask.distributed.LocalCluster(
            n_workers=N_WORKERS,
            threads_per_worker=THREADS_PER_WORKER, 
            memory_limit=MEM_PER_WORKER
        )
        client=dask.distributed.Client(cluster)
        yield client
        
    except Exception as e:
        logger.error(f"Error in cluster operations: {str(e)}")
        raise
    finally:
        # Cleanup sequence
        try:
            if client is not None:
                logger.info("Closing client...")
                client.close()
                
            if cluster is not None:
                logger.info("Closing cluster...")
                cluster.close()
                
            # Force garbage collection
            gc.collect()
            
            # Wait a moment for resources to be released
            time.sleep(2)
            
            # Log memory usage after cleanup
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            logger.info(f"Memory usage after cleanup: {memory_info.rss / 1024 / 1024:.2f} MB")
            
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {str(cleanup_error)}")    

def create_scl_mask(scl):
    valid_classes = [4, 5, 6, 11]  # vegetation, bare soil, water, snow and ice
    mask = scl.isin(valid_classes)
    return mask

def mask_scene(dataset):
    
    mask = create_scl_mask(dataset.scl)
    masked_dataset = dataset.where(mask)

    return masked_dataset, mask    

def select_best_scenes(dataset,n):
    """Select n best scenes based on percentage of valid pixels obtained after masking"""
    
    masked_dataset, mask = mask_scene(dataset)

    valid_percentage = (xr.where(mask,1,np.nan).count(dim=["x", "y"])/(mask.shape[-1]*mask.shape[-2])*100)

    # Select top n dates
    best_dates = (valid_percentage
        .sortby(valid_percentage, ascending=False)
        .head(n)
        .time)

    time_span = str( (best_dates.values.max()-best_dates.values.min()).astype('timedelta64[D]') )

    return masked_dataset.drop_vars(BANDS_DROP).sel(time=best_dates), time_span 

"""Create the series of tiles to process Iberian territory"""
def create_processing_tiles(size):
    spain = gpd.read_file('/Users/diegobengochea/git/iberian.carbon/data/SpainPolygon/gadm41_ESP_0.shp').to_crs(epsg='25830').iloc[0].geometry
    spain = odc.geo.geom.Geometry(spain,crs='EPSG:25830')
    spain = odc.geo.geobox.GeoBox.from_geopolygon(spain,resolution=10)

    # Divide the full geobox in Geotiles of 80 km for processing
    geotiles_spain = odc.geo.geobox.GeoboxTiles(spain,(size,size))
    geotiles_spain = [ geotiles_spain.__getitem__(tile) for tile in geotiles_spain._all_tiles() ]
    return geotiles_spain


def create_processing_list(size, years):
    return list(itertools.product(create_processing_tiles(size), years))

def get_tile_bounding_box(tile):
    bbox = tile.boundingbox.to_crs('EPSG:4326')
    bbox = (bbox.left,bbox.bottom,bbox.right,bbox.top)
    return bbox

"""Retrieve scenes"""

def search_catalog(catalog,bbox,time_range,cloud_threshold):
    search = catalog.search(
        collections = ['sentinel-2-l2a'],
        bbox = bbox, 
        datetime = time_range,
        query = [f'eo:cloud_cover<{cloud_threshold}']
    )

    item_collection = search.item_collection()
    
    if len(item_collection)<MIN_N_ITEMS and cloud_threshold<MAX_CLOUD_THRESHOLD:
        return search_catalog(catalog,bbox,time_range,cloud_threshold+10)
    else:
        return item_collection 

def load_dataset(item_collection, tile, bands, chunk_size):
    dataset = odc.stac.load(
        item_collection,
        bands = bands,
        geobox = tile,
        chunks = {'x':chunk_size,'y':chunk_size},
        groupby = 'solar_day',
        resampling = 'bilinear'
    )
    return dataset

def create_dataset(catalog,tile,time_range):
    bounding_box = get_tile_bounding_box(tile)
    item_collection = search_catalog(catalog,bounding_box,time_range,CLOUD_THRESHOLD)
    dataset = load_dataset(item_collection,tile,BANDS,CHUNK_SIZE)    
    return dataset

"""Saving"""
def save_mosaic(path):
    mosaic.rio.to_raster(
        path,
        tags = mosaic.attrs,
        **{'compress': 'lzw'},
        tiled = True,
        lock=dask.distributed.Lock('rio',client=client)
    )

if __name__ == '__main__':

    savedir = '/Users/diegobengochea/git/iberian.carbon/data/S2_summer_mosaics/'
    if not Path(savedir).exists():
        Path(savedir).mkdir(parents=True)

    catalog = Client.open(STAC_URL) 

    processing_list = create_processing_list(TILE_SIZE,YEARS)

    try: 
        for i,(tile, year) in enumerate(processing_list):

            bounding_box = get_tile_bounding_box(tile)
            resolution = abs(int(tile.resolution.x))

            north = int(np.round(bounding_box[3]))
            if bounding_box[0]>0:
                east = int(np.round(bounding_box[0]))
                savepath = f'{savedir}S2_summer_mosaic_{year}_N{north}_E{east}_{resolution}m.tif'
            else:
                west = - int(np.round(bounding_box[0]))
                savepath = f'{savedir}S2_summer_mosaic_{year}_N{north}_W{west}_{resolution}m.tif'
            

            if not Path(savepath).exists():

                logger.info(f'Found a non computed tile for year {year} in region {bounding_box}')
                try:
                    time_range = f'{year}-0{MIN_MONTH}-01/{year}-0{MAX_MONTH}-01'
                    with setup_optimized_cluster() as client:

                        logger.info(f"Cluster dashboard: {client.dashboard_link}")
                        configure_rio(cloud_defaults=True, client = client)
                        logger.info(f'Starting processing of mosaic {i+1}, {len(processing_list)-i} remaining')
                        try:
                            logger.info('Attempting to process')
                            dataset = create_dataset(catalog,tile,time_range)
                            mosaic = process_mosaic(dataset,year,time_range).compute()

                            logger.info(f'Mosaic {i+1} processed, {len(processing_list)-i-1} remaining')
                        
                            save_mosaic(savepath)
                            mosaic.close()

                        except Exception as process_error:
                            logger.error(f"Error processing year {year}: {str(process_error)}")
                            continue
                    
                        client.restart()
                
                except Exception as cluster_error:
                    logger.error(f"Cluster error for year {year} and tile {bounding_box}: {str(cluster_error)}")
                    continue

                gc.collect()

                memory_usage = psutil.virtual_memory()
                logger.info(f"System memory usage: {memory_usage.percent}%")

                time.sleep(5)

    except Exception as e:
        logger.error(f"Main process error: {str(e)}")
    finally:
        # Final cleanup
        gc.collect()
        logger.info("Processing completed")
            
