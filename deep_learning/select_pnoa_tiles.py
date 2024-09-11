import rasterio
from pathlib import Path
from shapely import geometry, intersects
import geopandas as gpd
import re
from rasterio.merge import merge
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
import numpy  as np

sentinel_path = '/Users/diegobengochea/git/iberian.carbon/data/LightningDataModule_Data/'

pnoa_path = '/Users/diegobengochea/git/iberian.carbon/Entornos_Vuelo_LiDAR/2_Cobertura/'
pnoa_datadir =  '/Users/diegobengochea/PNOA2/PNOA2_LIDAR_VEGETATION/'

target_pnoa_datadir = '/Users/diegobengochea/git/iberian.carbon/data/training_pnoa_tiles/'

fileids = set()
for year in [2020,2021]:
   for sentinel_tile in Path(sentinel_path).rglob('*{}*SWIR.tif'.format(year)):
        with rasterio.open(sentinel_tile)  as src_sentinel:
            
            bounds_sentinel = src_sentinel.bounds
            polygon_sentinel = geometry.box(minx=bounds_sentinel.left, maxx=bounds_sentinel.right, miny=bounds_sentinel.bottom, maxy=bounds_sentinel.top)

            tile_list = []
            
            for pnoa_polygons in Path(pnoa_path).rglob('*.shp'):
                # Each shapefile contains all rectangular polygons of the area sampled by the plane.

                utm = re.findall('_HU(..)_',str(pnoa_polygons))[0]
                pnoa_gdf = gpd.read_file(pnoa_polygons)
                intersecting_tiles = pnoa_gdf.intersects(polygon_sentinel)

                # Select only the polygons that intersect the sentinel tile
                pnoa_intersecting = pnoa_gdf[intersecting_tiles]
                #  Filter by year
                pnoa_intersecting = pnoa_intersecting[ pnoa_intersecting.FECHA == str(year)]
                # Add file names of the PNOA files to intersect.
                selected_files = pnoa_intersecting['PATH'].apply(lambda x: '{}{}'.format(pnoa_datadir,x.split('/')[-1]) ).to_list()


                #selected_dict = {re.findall('NDSM-VEGETACION-(?:...)-(.*)-COB2.tif',f)[0] : f for f in selected_files}
                #selected_dict = { fid : selected_dict[fid] for fid in selected_dict if not fid in fileids }
                # remove redundant files
                fileids.update((selected_files, year))

                # Create dict with new name that includes year of sampling!!!!!!! Then resave rasters by renaming. 

selected_new_fnames = [ (x, '{}/PNOA_{}_{}'.format(target_pnoa_datadir, x[1], re.findall('.*/(NDSM-.*)',x[0]))) for x in fileids ]
print(selected_new_fnames[0])

for ref, target in selected_new_fnames:
    with rasterio.open(ref) as src: 
        image = src.read(1)

        with rasterio.open(
            target,
            mode="w",
            driver="GTiff",
            height=image.shape[-2],
            width=image.shape[-1],
            dtype=np.float32,
            count=1,
            nodata=float(src.nodata),
            crs="EPSG:258{}".format(utm),
            transform=src.transform,
            compress='lzw'
        ) as new_dataset:
            new_dataset.write(image[0,:,:], 1) 
        

             
# if len(merging_dict)>0:
#     esa_id = re.findall('v.00_(.*)_SWIR.tif',str(sentinel_tile))[0]
#     merge_name = '/Users/diegobengochea/PNOA2_merged/PNOA_{}_H{}_{}_NDSM.tif'.format(year,utm,esa_id)
#     image, transform = merge( list(merging_dict.values()) )
#     with rasterio.open(
#         merge_name,
#         mode="w",
#         driver="GTiff",
#         height=image.shape[-2],
#         width=image.shape[-1],
#         dtype=np.float32,
#         count=1,
#         nodata=-32767.0,
#         crs="EPSG:258{}".format(utm),
#         transform=transform,
#         compress='lzw'
#     ) as new_dataset:
#         new_dataset.write(image[0,:,:], 1) 
                        



