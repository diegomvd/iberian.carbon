import geopandas as gpd
import pandas as pd
from pathlib import Path

forest_types_all= set()

mfe_dir = '/Users/diegobengochea/git/iberian.carbon/data/MFESpain/'
for mfe_file in Path(mfe_dir).glob('dissolved_by_form_MFE_*.shp'):
    forest_types = gpd.read_file(mfe_file).FormArbol.unique()
    forest_types_all.update(forest_types)


    
df = pd.DataFrame({'ForestType':list(forest_types_all)})
df.to_csv('all_forest_types.csv')
