import pandas as pd 
import geopandas as gpd
from geoutils import add_feature
from pathlib import Path

ifn4_biomass_dir = 'ifn4_biomass/'
ifn4_groups_utm = Path(ifn4_biomass_dir).glob('*.shp')

pnoa_file_info = pd.read_csv('PNOA2_flights/maps_year_table.csv')

for fname in ifn4_groups_utm:

    ifn4 = gpd.read_file(fname)
    crs = str(ifn4.crs)
    utm = crs[-2:]

    pnoa_info_filtered = pnoa_file_info[pnoa_file_info['utm'] == int(utm)]

    for year in pnoa_info_filtered['years'].unique():

        pnoa_info_filtered_year = pnoa_info_filtered[pnoa_info_filtered['years']==year]
        ifn4_year = ifn4[ifn4['year']==year]

        if len(ifn4_year.index)==0:
            continue

        ifn4_year_canopy = ifn4_year.copy()
        for index, row in pnoa_info_filtered_year.iterrows():
            fstem = row['file']
            path = 'PNOA2_LiDAR_Vegetation_25m/{}.tif'.format(fstem)

            ifn4_year_canopy = add_feature(ifn4_year_canopy,'h_{}'.format(index),path)    
        
        ifn4_year_canopy['height'] = ifn4_year_canopy.filter(like='h_').apply(lambda row: row.tolist(), axis=1)
        print(ifn4_year_canopy['height'].unique())
        
    

