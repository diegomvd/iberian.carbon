import rasterio
import numpy as np
from pathlib import Path 
import glob
import re
import pandas as pd

predictions_path = '/Users/diegobengochea/git/iberian.carbon/deep_learning/AGBD_predictions_CNIG/'
# predictions_path = '/home/diego/git/iberian.carbon/predictions_AGBD'

years = {re.findall(r'AGBD_(.*)_N',fname.stem)[0] for fname in Path(predictions_path).glob('*.tif')}

agb = pd.DataFrame()

for year in years:
    for fname in Path(predictions_path).glob('*.tif'):
        if year in fname.stem:
            with rasterio.open(fname) as src: 
                image=src.read(1)
                image=np.where(image==src.nodata,np.nan,image)
                agb_year += np.nansum(image)*0.01
    agb_year = agb_year*np.power(10.,-6.)            
    row = pd.DataFrame({'Year':[year], 'AGB(Gt)' : agb_year}) 
    agb = pd.concat([agb,row],axis='rows')           

agb.to_csv('AGB_stocks_N44.csv', index = False)

    # year_dict[year] = flist 

# year_dict = {}
# for year in years:
#     flist = []
#     for fname in Path(predictions_path).glob('*.tif'):
#         if year in fname.stem:
#             flist.append(fname)
#     year_dict[year] = flist 



# smallest_set_size = np.min({len(fnames) for fnames in year_dict.values()})

# years_with_excedent_tiles = [year for year in years_dict if len(years_dict[year])>smallest_set_size]
# years_with_minimum_tiles = [year for year in years_dict if len(years_dict[year])>smallest_set_size]

# for year in year_dict:
#     if len(year_dict)>smallest_set_size>


# print(smallest_set_size)           



# tiles = {re.findall(r'_(N.*)',fname.stem)[0] for fname in Path(predictions_path).glob('*.tif')}

# tile_dict = {}
# for tile in tiles:
#     flist = []
#     for fname in Path(predictions_path).glob('*.tif'):
#         if tile in fname.stem:
#             flist.append(fname)
#     tile_dict[tile] = flist 

# print(tile_dict)        