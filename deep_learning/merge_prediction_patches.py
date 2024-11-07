import rasterio 
import numpy as np 
from rasterio.merge import merge 
from pathlib import Path
import re
import odc.geo

target = 'canopy_height_full_data'
# target = 'landcover_segmentation'

def convert_years(year):
    match year:
        case 1514761200:
            return 2018
        case 1546297200:
            return 2019
        case 1577833200:
            return 2020
        case 156:
            return 2021
        case 157:
            return 2022
        case 158:
            return 2023
        case 159:
            return 2024                        

# Build a grid with target tiles

spain = gpd.read_file('/Users/diegobengochea/git/iberian.carbon/data/SpainPolygon/gadm41_ESP_0.shp')
# Reproject to UTM30 EPSG:25830
spain = spain.[['geometry','COUNTRY']].to_crs(epsg='25830')
# Add CRS information to shapely polygon
geometry_spain = odc.geo.geom.Geometry(spain.geometry[0],crs='EPSG:25830')
# Create a GeoBox for all continental Spain with a 10 meters resolution 
geobox_spain = odc.geo.geobox.GeoBox.from_geopolygon(geometry_spain,resolution=10) # The resolution here is irrelevant since the Spain GeoBOX is too large to make queries, adn thus cannot be used as intersect. Only used to create tiles of suitable shape 20km2

# Divide the full geobox in Geotiles of 120kmx120km
geotiles_spain = odc.geo.geobox.GeoboxTiles(geobox_spain,(12000,12000))
geotiles_spain = [ geotiles_spain.__getitem__(tile) for tile in geotiles_spain._all_tiles() ]



# This is useful if all prediction would be ready, in the meanwhile select manually the years to merge
# all_files = Path(path).glob('*.tif')
# years = {re.search(r'_mint_(*).tif',fname.stem).group() for fname in all_files}

years = [1514761200,1546297200]
args_list = list(itertools.product(geotiles_spain, years))

predictions = f"/Users/diegobengochea/git/iberian.carbon/deep_learning/predictions_{target}_CNIG/0/"

for args in args_list:

    # Buffer the tile to prevent stitching artifacts
    original_tile = args[0]
    tile = original_tile.buffered(2650) # in meters
    year = args[1]

    tile_bbox = tile.boundingbox
    tile_shapely_box = shapely.box(
        tile_bbox.left,
        tile_bbox.bottom,
        tile_bbox.right,
        tile_bbox.top)

    tile_bbox_latlon = tile_bbox.to_crs('EPSG:4326')
    lon = tile_bbox_latlon.left 
    lat = tile_bbox_latlon.top   

    all_files = [file for file in Path(predictions).glob(f'*{year}.tif') ] 

    for fname in all_files:
        with rasterio.open(fname) as src:
            bounds = src.bounds
            prediction_shapely_box = shapely.box(
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top)
            if shapely.intersects(tile_shapely_box,prediction_shapely_box)
                files_to_merge.append(fname)
            else:
                continue 

            
    image_sum, transform_sum = merge(files_to_merge, method = 'sum')
    image_count, transform_count = merge(files_to_merge, method = 'count')
    # Final image averaging overlapping tile buffers
    image = image_sum/image_count

    year_to_print = convert_years(year)
    lat_to_print=np.round(lat,0)
    if lon > 0
        lon_to_print = np.round(lon,0)
        savepath = f"/Users/diegobengochea/git/iberian.carbon/deep_learning/predictions_{target}_CNIG/merged_120km/{target}_{year_to_print}_N{lat_to_print}_E{lon_to_print}.tif" 
    else:
        lon_to_print = np.round(-lon,0)
        savepath = f"/Users/diegobengochea/git/iberian.carbon/deep_learning/predictions_{target}_CNIG/merged_120km/{target}_{year_to_print}_N{lat_to_print}_W{lon_to_print}.tif" 

    with rasterio.open(
            savepath,
            mode="w",
            driver="GTiff",
            height=image.shape[-2],
            width=image.shape[-1],
            count=1,
            dtype= 'float32',
            crs="epsg:25830",
            transform=original_tile.transform,
            nodata=-1.0,
            compress='lzw'    
        ) as new_dataset:
            new_dataset.write(image[0,:,:], 1)
            new_dataset.update_tags(DATE = year_to_print)

