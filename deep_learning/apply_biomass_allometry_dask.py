import pandas as pd 
import geopandas as gpd
import rasterio 
import numpy as np 
import rasterio.mask
from pathlib import Path
import re
import itertools
import shapely
import logging
import dask.distributed
import dask.array as da
from dask.diagnostics import ProgressBar
import gc
import os
import psutil
import time
from contextlib import contextmanager
import xarray as xr
import rioxarray as rio
import random
import dask.dataframe as dd
# import xarray.ufuncs as xu

ALLOMETRIES_DIR = '/Users/diegobengochea/git/iberian.carbon/data/stocks_NFI4/H_AGB_Allometries_Tiers.csv'
FOREST_TYPE_DIR = '/Users/diegobengochea/git/iberian.carbon/data/stocks_NFI4/Forest_Types_Tiers.csv'
CANOPY_HEIGHT_DIR = '/Users/diegobengochea/git/iberian.carbon/deep_learning/canopy_height_predictions/merged_240km/'
MFE_DIR = '/Users/diegobengochea/git/iberian.carbon/data/MFESpain/'
HEIGHT_THRESHOLD = 0.5 #meters
TIER_NAMES = {0:'Dummy',1:'Clade',2:'Family',3:'Genus',4:'ForestTypeMFE'}

N_WORKERS=2
THREADS_PER_WORKER=3
MEM_PER_WORKER='86Gb'
CHUNK_SIZE = 800

###################################################################

def build_savepath(fname):
    stem = fname.stem
    specs = re.findall(r'canopy_height_(.*)',stem)[0]
    savepath = f'/Users/diegobengochea/git/iberian.carbon/data/predictions_AGBD/AGBD_{specs}.tif'
    return savepath

def load_canopy_height_image(src):
    
    bounds = src.bounds
    box = shapely.box(
        bounds.left,
        bounds.bottom,
        bounds.right,
        bounds.top)

    ds_canopy_height = rio.open_rasterio(src,chunks={'band': 1, 'x': CHUNK_SIZE, 'y': CHUNK_SIZE})
    ds_canopy_height = xr.where(ds_canopy_height == ds_canopy_height.rio.nodata,np.nan,ds_canopy_height)                                   
    ds_canopy_height = xr.where(ds_canopy_height<HEIGHT_THRESHOLD,0.,ds_canopy_height).rio.write_crs('epsg:25830',inplace=True)

    return ds_canopy_height, box

def update_tiers(tier,tier_names):
    old_tier_name = tier_names[tier]
    new_tier = tier - 1
    new_tier_name = tier_names[new_tier]

    return new_tier, old_tier_name, new_tier_name

def update_forest_type(forest_types, forest_type, old_tier_name, new_tier_name):

    new_forest_type = forest_types.reset_index().set_index(old_tier_name).loc[forest_type][new_tier_name]
    if not isinstance(new_forest_type,str):
        new_forest_type = forest_types.reset_index().set_index(old_tier_name).loc[forest_type].iloc[0][new_tier_name]
    return new_forest_type

@dask.delayed
def select_best_allometry(forest_type_input,forest_types,tier_input,allometries,tier_names):
    tier = tier_input
    forest_type = forest_type_input

    while tier>=0:  
        try:
            allom_row = allometries[allometries.Tier==tier].loc[forest_type]
            tier = -1
        except:
            try:
                tier, old_tier_name, new_tier_name = update_tiers(tier,tier_names)
                
                forest_type = update_forest_type(forest_types,forest_type,old_tier_name,new_tier_name)
                
            except Exception as e:
                logger.error(f'Error in allometry selection: {str(e)}, returning general allometry.')
                raise
                # try:
                #     allom_row = allometries[allometries.Tier==0].loc['General']
                # except Exception as e:
                #     logger.error(f'Error retrieving general allometry: {str(e)}')
                #     raise


    intercept_mean = da.exp(allom_row.Intercept_mean)
    slope_mean = allom_row.Slope_mean
    
    return intercept_mean, slope_mean


@dask.delayed
def apply_allometry(intercept, slope, output_ds, input_ds, mask):

    output_ds_updated = xr.where(
        mask,
        intercept*(input_ds**slope),
        output_ds
    )

    return output_ds_updated

@dask.delayed
def create_forest_mask(geometry_iter, input_ds):

    forest_type_clip = input_ds.rio.clip(
        geometries = geometry_iter,
        drop = False,
        invert = False
    ).rio.write_crs('epsg:25830',inplace=True)
    
    forest_type_mask = xr.where(forest_type_clip==forest_type_clip.rio.nodata,False,True).rio.write_crs('epsg:25830',inplace=True)

    del forest_type_clip
    gc.collect()

    return forest_type_mask


def calculate_aboveground_biomass(ds_canopy_height,canopy_height_box,allometries_df,forest_types_table,tier_names, mfe_paths):
    
    ds_agb_mean = xr.full_like(ds_canopy_height,0.0,chunks={'band': 1, 'x': CHUNK_SIZE, 'y': CHUNK_SIZE}).rio.write_crs('epsg:25830',inplace=True)
    
    for mfe_fname in mfe_paths:

        mfe_contour = gpd.read_file(f'{MFE_DIR}dissolved_{mfe_fname.stem[-6:]}.shp').iloc[0]['geometry']

        if shapely.intersects(canopy_height_box,mfe_contour):

            #del mfe_contour
            #gc.collect()

            mfe = gpd.read_file(mfe_fname).set_index('FormArbol')

            for ix, row in mfe.iterrows():
            
                if shapely.intersects(canopy_height_box,row.geometry):

                    if not ix == 'No arbolado':

                        coeffs = select_best_allometry(ix,forest_types_table,4,allometries_df,tier_names)
                        intercept_mean = coeffs[0]
                        slope_mean = coeffs[1]

                        try:
                            
                            forest_type_mask = create_forest_mask([row.geometry],ds_canopy_height)

                            ds_agb_mean = apply_allometry(intercept_mean,slope_mean,ds_agb_mean,ds_canopy_height,forest_type_mask)
                        
                        except Exception as e:
                            logger.error(f'Error applying allometries {str(e)}')
                            raise
            #del mfe 
            #gc.collect()

    return ds_agb_mean

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

def process_tile(fname,allom_table,forest_type_table):

    da_allom_table = dd.from_pandas(allom_table,npartitions=1)
    da_forest_type_table = dd.from_pandas(forest_type_table, npartitions=1)

    mfe_files = Path(MFE_DIR).glob('dissolved_by_form_MFE*.shp')

    with rasterio.open(fname) as src:
        
        dataset, box = load_canopy_height_image(src)  
        logger.info('Canopy height dataset was lazy loaded')

        tier_names = TIER_NAMES

        agb_dataset = calculate_aboveground_biomass(dataset,box,allom_table,forest_type_table,tier_names,mfe_files)
        logger.info('AGB calculation graph built')

     #   del dataset
      #  gc.collect()

        agb_dataset = agb_dataset.fillna(src.nodata).astype('float32')
        agb_dataset.rio.write_nodata(src.nodata, inplace=True)
        
    return agb_dataset

def save_tile(tile,path):
    tile.rio.to_raster(
        path,
        tags = tile.attrs,
        **{'compress': 'lzw'},
        tiled = True,
        lock=dask.distributed.Lock('rio',client=client)
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    logger.info('Loading fitted allometries by forest type tiers.')

    allometries = pd.read_csv(ALLOMETRIES_DIR).set_index('ForestType')
    
    logger.info('Loading forest type tiers.')

    forest_types = pd.read_csv(FOREST_TYPE_DIR)
    forest_types['Dummy']='General'

    #forest_types = dask.delayed(forest_types_c)
    
    files = [path for path in Path(CANOPY_HEIGHT_DIR).glob('*.tif')]
    random.shuffle(files)
    
    try:
        for i,fname in enumerate(files):

            logger.info(f'Progress: {i/len(files)*100}%.')
            
            savepath = build_savepath(fname)
            
            if not Path(savepath).exists():
            
                logger.info(f'Found non-processed tile {Path(savepath).stem}')

                try:
                    with setup_optimized_cluster() as client:

                        logger.info(f"Cluster dashboard: {client.dashboard_link}")
                        #configure_rio(cloud_defaults=True, client = client)
                        logger.info(f'Starting processing of tile {i+1}, {len(files)-i} remaining')

                        try:
                            
                            logger.info('Loading canopy height data and calculating AGB')

                            agb_dataset = process_tile(fname,allometries,forest_types)

                      #      if not processed:
                       #         logger.warning(f'There were no intersecting MFE forest polygons with this tile. This should not happen, projections might not be matching, or MFE shapefiles might be missing from the source directory. Alternatively, canopy height tiles might be outside of the Iberian Peninsula, which is currently not supported. Lastly, verify that tiles are not entirely over water bodies.')
                      #          continue
                            
                            logger.info('Starting graph computation.')
                            agb_image = agb_dataset.compute()

                        except Exception as process_error:
                            logger.error(f"Error processing tile {Path(fname).stem}: {str(process_error)}")
                            continue    

                        logger.info(f'Finished processing. Saving resulting dataset to {savepath}.')
                        try:
                            save_tile(agb_image,savepath)        

                        except Exception as e:
                            logger.error(f'Error saving dataset: {str(e)}')
                            continue
                        
                        del agb_image
                        client.restart()

                except Exception as cluster_error:
                    logger.error(f"Cluster error for year {year} and tile {bounding_box}: {str(cluster_error)}")
                    continue

                gc.collect()    
    except Exception as e:
        logger.error(f"Main process error: {str(e)}")
    finally:
        # Final cleanup
        gc.collect()
        logger.info("Processing completed")

    logger.info('Processing done. Finishing program.')        


