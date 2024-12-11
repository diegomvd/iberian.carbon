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

ALLOMETRIES_DIR = '/Users/diegobengochea/git/iberian.carbon/data/stocks_NFI4/H_\
AGB_Allometries_Tiers.csv'
FOREST_TYPE_DIR = '/Users/diegobengochea/git/iberian.carbon/data/stocks_NFI4/F\
orest_Types_Tiers.csv'
CANOPY_HEIGHT_DIR = '/Users/diegobengochea/git/iberian.carbon/deep_learning/canopy_height_predictions/merged_240km/'
MFE_DIR = '/Users/diegobengochea/git/iberian.carbon/data/MFESpain/'
HEIGHT_THRESHOLD = 0.5 #meters
TIER_NAMES = {0:'Dummy',1:'Clade',2:'Family',3:'Genus',4:'ForestTypeMFE'}

###################################################################

def build_savepath(fname):
    stem = fname.stem
    specs = re.findall(r'canopy_height_(.*)',stem)[0]
    savepath = f'/Users/diegobengochea/git/iberian.carbon/data/predictions_AGBD/AGBD_{specs}.tif'
    return savepath

def load_canopy_height_image(fname):
    with rasterio.open(fname) as src:
        bounds = src.bounds
        box = shapely.box(
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top)

        image = src.read(1)
        image = np.where(image==src.nodata,np.nan,image)

        # Filter out values lower than 0.5 meters                                   
        image = np.where(image<HEIGHT_THRESHOLD,0.,image)
    return src, image, box

def select_best_allometry(forest_type,tier,allometries):
    if forest_type == 'No arbolado':
        allom_row = pd.DataFrame({
            'Intercept_mean':[0.0],
            'Slope_mean':[0.0],
            'Intercept_q05':[0.0],
            'Slope_q05':[0.0],
            'Intercept_q95':[0.0],
            'Slope_q95':[0.0],
            
        })
    else:
        try:
            allom_row = allometries[allometries.Tier==tier].loc[forest_type]
            #logger.info(f'Found fitted allometry for tier {tier}.')
        except:
            try:
                old_tier_name = TIER_NAMES[tier]
                tier-=1
                new_tier_name = TIER_NAMES[tier]

                new_forest_type = forest_types.reset_index().set_index(old_tier_name).loc[forest_type][new_tier_name]
                if not isinstance(new_forest_type,str):
                    try:
                        new_forest_type = forest_types.reset_index().set_index(old_tier_name).loc[forest_type].iloc[0][new_tier_name]
                    except:
                        raise
                allom_row = select_best_allometry(new_forest_type,tier,allometries)
            except Exception as e:
                logger.error(f'Error in allometry selection: {str(e)}')
                logger.info('Returning general allometry.')
                try:
                    allom_row = allometries[allometries.Tier==0].loc['General']
                except Exception as e:
                    logger.error(f'Error retrieving general allometry: {str(e)}')
                    raise

    return allom_row
    
def power_law(x,a,b):
    return a*np.power(x,b)

def parallel_where(forest_type, canopy_height_image, allometries, image_mean, image_q05, image_q95):
    # Square chunks for balanced memory and computation
    chunk_size = 6000  # 6000x6000 pixels
    
    # Convert to Dask arrays with square chunking
    forest_type_da = da.from_array(forest_type, chunks=(chunk_size, chunk_size))
    canopy_height_da = da.from_array(canopy_height_image, chunks=(chunk_size, chunk_size))
    image_mean_da = da.from_array(image_mean, chunks=(chunk_size, chunk_size))
    image_q05_da = da.from_array(image_q05, chunks=(chunk_size, chunk_size))
    image_q95_da = da.from_array(image_q95, chunks=(chunk_size, chunk_size))
    
    # Create Dask client
    client = dask.distributed.Client(
        n_workers=9,
        threads_per_worker=2,
        memory_limit='21.3GB'
    )
    
    # Computation with progress tracking
    with ProgressBar():
        future_mean = da.where(
            forest_type_da,
            power_law(canopy_height_da, allometries['mean'][0], allometries['mean'][1]),
            image_mean_da
        )
        
        future_q05 = da.where(
            forest_type_da,
            power_law(canopy_height_da, allometries['q05'][0], allometries['q05'][1]),
            image_q05_da
        )
        
        future_q95 = da.where(
            forest_type_da,
            power_law(canopy_height_da, allometries['q95'][0], allometries['q95'][1]),
            image_q95_da
        )
        
        # Compute results
        result_mean, result_q05, result_q95 = dask.compute(
            future_mean, future_q05, future_q95
        )
    
    # Close Dask client
    client.close()
    
    return result_mean, result_q05, result_q95
    

def calculate_aboveground_biomass(canopy_height_src,canopy_height_image,canopy_height_box,allometries_df):

    intersected_any = False
    for mfe_fname in Path(MFE_DIR).glob('dissolved_by_form_MFE*.shp'):
        
        mfe_contour = gpd.read_file(f'{MFE_DIR}dissolved_{mfe_fname.stem[-6:]}.shp')

        #logger.info(f'Processing CA {mfe_fname.stem[-2:]}.')

        if shapely.intersects(canopy_height_box,mfe_contour.iloc[0]['geometry']):

            processed_cover = 0
            
            mfe = gpd.read_file(mfe_fname)
            mfe = mfe.set_index('FormArbol')

            for ix, row in mfe.iterrows():
            
                if shapely.intersects(canopy_height_box,row.geometry):

                    if not intersected_any:
                        intersected_any = True
                        image_mean = np.full(canopy_height_image.shape,0)
                        image_q05 = np.full(canopy_height_image.shape,0)
                        image_q95 = np.full(canopy_height_image.shape,0)
                        
                    logger.info(f'Processing forest type {ix}.')
                    if not ix == 'No arbolado':

                        forest_type,_,_=rasterio.mask.raster_geometry_mask(
                            canopy_height_src,
                            [row.geometry],
                            crop=False,
                            invert=True)

                        processed_cover += np.count_nonzero(forest_type)/forest_type.size*100
                        logger.info(f'Processed cover {processed_cover}%')
                        
                        allom_row = select_best_allometry(ix,max(TIER_NAMES.keys()),allometries_df)
                        try:
                            intercept_mean =np.exp(allom_row.Intercept_mean)
                            slope_mean = allom_row.Slope_mean
                            intercept_q05 = np.exp(allom_row.Intercept_q05)
                            slope_q05 = allom_row.Slope_q05
                            intercept_q95 = np.exp(allom_row.Intercept_q95)
                            slope_q95 = allom_row.Slope_q95
                        
                            allometries = {
                                'mean':(intercept_mean,slope_mean),
                                'q05':(intercept_q05,slope_q05),
                                'q95':(intercept_q95,slope_q95)
                            }

    
                            image_mean = np.where(
                                forest_type,
                                power_law(canopy_height_image, allometries['mean'][0], allometries['mean'][1]),
                                image_mean
                            )

                            image_q05 = np.where(
                                forest_type,
                                power_law(canopy_height_image, allometries['q05'][0], allometries['q05'][1]),
                                image_q05
                            )

                            image_q95 = np.where(
                                forest_type,
                                power_law(canopy_height_image, allometries['q95'][0], allometries['q95'][1]),
                                image_q95
                            )
                        
                        except Exception as e:
                            logger.error(f'Error applying allometries {str(e)}')
                            raise

    logger.info('Finished applying allometries. Stacking images.')            
    try:            
        agb_image = np.stack([image_mean,image_q05,image_q95])
    except Exception as e:
        logger.error(f'Error stacking images: {str(e)}')
        

    return agb_image, intersected_any

    

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info('Loading fitted allometries by forest type tiers.')

allometries = pd.read_csv(ALLOMETRIES_DIR)
allometries = allometries.set_index('ForestType')

logger.info('Loading forest type tiers.')

forest_types = pd.read_csv(FOREST_TYPE_DIR)
forest_types['Dummy']='General'

if __name__ == '__main__':

    files = [path for path in Path(CANOPY_HEIGHT_DIR).glob('*.tif')]
    for i,fname in enumerate(files):

        logger.info(f'Processing tile {i+1} out of {len(files)}. Progress: {i/len(files)*100}%.')
        savepath = build_savepath(fname)
        if not Path(savepath).exists():
            logger.info(f'Processing target {Path(savepath).stem}. Loading image...')
            try:
                src, image, box = load_canopy_height_image(fname)
                try:
                    logger.info('...done. Proceeding to biomass calculation.')
                    image, processed = calculate_aboveground_biomass(src,image,box,allometries)
                    logger.info(f'Processed is {processed}')
                    logger.info(f'Image shape is {image.shape}')
                    try:
                        if processed:
                            logger.info(f'Finished processing. Saving resulting dataset to {savepath}.')
                            image = np.where(np.isnan(image),src.nodata,image)
                            try:
                                with rasterio.open(
                                        savepath,
                                        mode="w",
                                        driver="GTiff",
                                        height=image.shape[-2],
                                        width=image.shape[-1],
                                        count=3,
                                        dtype= 'float32',
                                        crs="epsg:25830",
                                        transform=src.transform,
                                        nodata=src.nodata,
                                        compress='lzw'
                                ) as new_dataset:
                                    new_dataset.write(image[:,:,:])
                                    #new_dataset.update_tags(DATE = year_to_print)                                                                                            
                            except Exception as e:
                                logger.error(f'Error saving dataset: {str(e)}')
                                continue
                        else:
                            logger.warning(f'There were no intersecting MFE forest polygons with this tile. This should not happen, projections might not be matching, or MFE shapefiles might be missing from the source directory. Alternatively, canopy height tiles might be outside of the Iberian Peninsula, which is currently not supported.')
                    except Exception as e:
                        logger.error(f'Error inputing no data {str(e)}')
                        continue
                except Exception as e:
                    logger.error(f'Failed at biomass calculation. Error {str(e)}')
                    continue
            except Exception as e:
                logger.error(f'Failed loading canopy height data. Error {str(e)}')
                continue

    logger.info('Processing done. Finishing program.')        


