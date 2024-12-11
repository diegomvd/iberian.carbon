import geopandas as gpd
import pandas as pd
from pathlib import Path
import re
import numpy as np
import rasterio.mask
from geoutils import add_feature
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ifn_biomass_fnames = Path('/Users/diegobengochea/git/iberian.carbon/data/stocks_NFI4/').glob('*foresttype_PNOAflight.shp')

logger.info('Starting sampling of PNOA LiDAR-derived vegetation height.')
height_biomass_df = pd.DataFrame()
for fname in ifn_biomass_fnames:

    utm = re.findall('ifn4_(.*)_biomass',fname.stem)[0]

    logger.info(f'Loading points in UTM {utm}.')
    
    ifn4 = gpd.read_file(fname)

    # for p in range(80,100,5):
    #     ifn4['Hp{}'.format(p)]=np.nan
    ifn4['Hmean']=np.nan
    # ifn4['Hmax']=np.nan

    logger.info('Starting iteration over points.')
    for ix, row in ifn4.iterrows():
        circle = [row.geometry.buffer(25)]

        mission = row.mission
        if mission=='1':
            pnoa_dir = '/Users/diegobengochea/Dropbox (Maestral)/Diego Bengochea/PNOA1_LIDAR_vegetation/'
        if mission=='2':
            pnoa_dir = '/Users/diegobengochea/AraujoLab Dropbox/Diego Bengochea/PNOA2_LIDAR_vegetation/'    
        pnoa_path = f'{pnoa_dir}{row.pnoa_file}'

        with rasterio.open(pnoa_path) as pnoa_src:

            pnoa_image, pnoa_trf = rasterio.mask.mask(pnoa_src, circle, crop=True)
            pnoa_image= np.where(pnoa_image==pnoa_src.nodata, np.nan, pnoa_image) 

            # for p in range(80,100,5):
            #     ifn4.at[ix,'Hp{}'.format(p)] = np.nanpercentile(pnoa_image,q=p)
            ifn4.at[ix,'Hmean']=np.nanmean(pnoa_image)
            # ifn4.at[ix,'Hmax']=np.nanmax(pnoa_image)  

    ifn4=ifn4.dropna()      

    logger.info(f'Resulting point layer has size {len(ifn4.index)}. Filtering relevant columns.')
    
    # ifn4 = add_feature(ifn4.to_crs("EPSG:4326"),"BIOME_NAME","ecoregions/Ecoregions2017.shp","Biome")
    ifn4 = add_feature(ifn4.to_crs("EPSG:4326"),"ECO_NAME","/Users/diegobengochea/git/iberian.carbon/ecoregions/Ecoregions2017.shp","Ecoregion")
    
    # columns = ['Hp{}'.format(p) for p in range(80,100,5)]
    columns=['Hmean']
    # columns.append('Hmean')
    # columns.append('Hmax')
    columns.append('AGB')
    columns.append('BGB')
    columns.append('Year')
    # columns.append('Type')
    columns.append('Ecoregion')
    columns.append('ForestType')
    columns.append('Index_1')
    
    tmp = ifn4[columns]   
    height_biomass_df = pd.concat([height_biomass_df,tmp],axis='rows')

logger.info('Processing done. Saving results...')    
height_biomass_df.to_csv('/Users/diegobengochea/git/iberian.carbon/data/stocks_NFI4/HeightBiomassTable.csv',index=False)
logger.info('...done.')
