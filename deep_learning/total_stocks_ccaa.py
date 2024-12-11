import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask
from pathlib import Path
import re
import gc
import logging
import shapely
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info('Load CCAA polygons')
ccaa_spain = gpd.read_file('/Users/diegobengochea/git/iberian.carbon/data/SpainPolygon/gadm41_ESP_1.shp').to_crs('epsg:25830')

logger.info('Build list of AGB tiles to process:')
agb_tiles_dir = '/Users/diegobengochea/git/iberian.carbon/data/predictions_AGBD/'
years = ['2017','2018','2019','2020','2021']
agb_tiles = [ (re.findall(r'AGBD_(\d{4})_',fname.stem)[0], fname) for fname in Path(agb_tiles_dir).glob('*.tif') if any(year in str(fname) for year in years)]

logger.info('Iterate over tiles.')
agb_stocks_ccaa = pd.DataFrame()
for year, tile in agb_tiles:

    logger.info(f'Processing tile {tile}. Reading dataset')
    with rasterio.open(tile) as src:
        bounds = src.bounds
        box = shapely.box(
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top
        )

        logger.info('Looking for intersecting CCAA')
        intersected_any = False
        for ix, row in ccaa_spain.iterrows():

            polygon = row.geometry
            ca_name = row.NAME_1

            if shapely.intersects(box,polygon):
                intersected_any = True

                logger.info(f'Intersected {ca_name}')

                agb_community_raster, _ = mask(src,[polygon],crop=True)

                # Deal with potential multiband with error margins in some tiles
                try:
                    # select only first band
                    agb_community_raster = agb_community_raster[0,:,:]
                except:
                    # If it is not multiband and it only has two dimensions just keep going
                    pass

                logger.info('Sum AGB stocks')
                agb_community_raster = np.where(agb_community_raster==src.nodata,np.nan,agb_community_raster)

                total_agb_community = np.nansum(agb_community_raster)*np.power(10.,-8.) # includes conversion to total AGB from AGBD in 100m2 and from tonnes to megatonnes.

                row = pd.DataFrame({'AGB(Mt)':[total_agb_community],'Year':[year],'CA':[ca_name],'Count':[1]})
                agb_stocks_ccaa = pd.concat([agb_stocks_ccaa,row],axis='rows')
                
                del agb_community_raster
                gc.collect()

        if not intersected_any:
            logger.warning('No intersecting CCAA found. This should not happen. Maybe CCAA CRS and tiles CRS do not match.')

agb_stocks_ccaa = agb_stocks_ccaa.groupby(['Year','CA']).agg({'AGB(Mt)':'sum','Count':'sum'}).reset_index()
agb_stocks_ccaa.to_csv(f'/Users/diegobengochea/git/iberian.carbon/data/Total_AGB_CCAA_{years[0]}-{years[-1]}.csv',index=False)
