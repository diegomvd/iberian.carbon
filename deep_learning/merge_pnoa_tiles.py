import rasterio
from pathlib import Path
from shapely import geometry, intersects
import geopandas as gpd
import re
from rasterio.merge import merge
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
import numpy  as np

sentinel_path = '/Users/diegobengochea/WorldCoverComposites/WORLDCOVER/'

pnoa_path = '/Users/diegobengochea/git/iberian.carbon/Entornos_Vuelo_LiDAR/2_Cobertura/'
pnoa_dataddir =  '/Users/diegobengochea/PNOA2_LIDAR_VEGETATION/'

for year in [2020,2021]:
    for sentinel_tile in Path(sentinel_path).rglob('*{}*SWIR.tif'.format(year)):
        print(sentinel_tile)
        
        with rasterio.open(sentinel_tile)  as src_sentinel:
            
            bounds_sentinel = src_sentinel.bounds
            polygon_sentinel = geometry.box(minx=bounds_sentinel.left, maxx=bounds_sentinel.right, miny=bounds_sentinel.bottom, maxy=bounds_sentinel.top)

            tile_list = []
            fileids = set()
            print('here')
            for pnoa_polygons in Path(pnoa_path).rglob('*.shp'):
                utm = re.findall('_HU(..)_',str(pnoa_polygons))[0]
                print('utm')
                pnoa_gdf = gpd.read_file(pnoa_polygons)
                intersecting_tiles = pnoa_gdf.intersects(polygon_sentinel)
                pnoa_intersecting = pnoa_gdf[intersecting_tiles]
                pnoa_intersecting = pnoa_intersecting[ pnoa_intersecting.FECHA == str(year)]
                merging_files = pnoa_intersecting['PATH'].apply(lambda x: '{}{}'.format(pnoa_dataddir,x.split('/')[-1]) ).to_list()


                merging_dict = {re.findall('NDSM-VEGETACION-(?:...)-(.*)-COB2.tif',f)[0] : f for f in merging_files}
                merging_dict = { fid : merging_dict[fid] for fid in merging_dict if not fid in fileids }
                fileids.update(merging_dict.keys())

                if len(merging_dict)>0:
                    esa_id = re.findall('v.00_(.*)_SWIR.tif',str(sentinel_tile))[0]
                    merge_name = '/Users/diegobengochea/PNOA2_merged/PNOA_{}_H{}_{}_NDSM.tif'.format(year,utm,esa_id)
                    image, transform = merge( list(merging_dict.values()) )
                    with rasterio.open(
                        merge_name,
                        mode="w",
                        driver="GTiff",
                        height=image.shape[-2],
                        width=image.shape[-1],
                        dtype=np.float32,
                        count=1,
                        nodata=-32767,
                        crs="EPSG:258{}".format(utm),
                        transform=transform,
                        compress='lzw'
                    ) as new_dataset:
                        new_dataset.write(image[0,:,:], 1) 
                



