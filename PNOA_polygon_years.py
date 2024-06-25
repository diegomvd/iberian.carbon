import geopandas as gpd
import pandas as pd
import rasterio
from pathlib import Path
import re
from shapely.geometry import box


pnoa2_ambito_dir = 'ambito_PNOA_2/'
pnoa2_ambito_subdir = Path(pnoa2_ambito_dir).glob('*')

polygons_year = gpd.GeoDataFrame()

for subdir in pnoa2_ambito_subdir:
    pnoa2_ambito_fnames = subdir.glob('*.shp')
    for fname in pnoa2_ambito_fnames:
        year_list = re.findall('_(20.*)',fname.stem)

        if len(year_list)==0:
            year = '201X'
        elif len(year_list[0])==4:
            year = year_list[0]
        else:
            year = re.findall('(.*)_',year_list[0])[0]

        polygon = gpd.read_file(fname)
        polygon['year'] = year
        polygon_reduced = polygon[['geometry','year']]
        try:
            polygon_reduced = polygon_reduced.to_crs('EPSG:25830')
        except:
            polygon_reduced = polygon_reduced.set_crs('EPSG:25830')
        polygons_year = pd.concat([polygons_year,polygon_reduced],axis='rows')  

if not Path('PNOA2_flights/').exists():
    Path('PNOA2_flights/').mkdir()
if not Path('PNOA2_flights/coverage_year_map.shp').exists():
    polygons_year.to_file('PNOA2_flights/coverage_year_map.shp',driver='ESRI Shapefile')


pnoa_25m_dir = 'PNOA2_LiDAR_Vegetation_25m/'
pnoa_fnames = Path(pnoa_25m_dir).glob('*')

pnoa2_maps_year = pd.DataFrame([])

for fname in pnoa_fnames:
    utm = re.findall('-H(.*)',fname.stem)[0][:2]

    with rasterio.open(fname) as src:
        bounds = src.bounds
        map_crs = src.crs

        polygons_year_reprojected = polygons_year.to_crs(map_crs)
        bbox_polygon = box(*bounds)
        
        year_list = []
        for index, row in polygons_year_reprojected.iterrows():
            pnoa_coverage_polygon = row['geometry']
            
            if bbox_polygon.intersects(pnoa_coverage_polygon):
                coverage_year = row['year']
                year_list.append(coverage_year)

        this_df = pd.DataFrame({'file':fname.stem,'years':year_list,'utm':utm})        
        pnoa2_maps_year = pd.concat([pnoa2_maps_year,this_df],axis='rows')
        
pnoa2_maps_year.reindex().to_csv('maps_year_table.csv',index=False)        
print(pnoa2_maps_year)        
            


    