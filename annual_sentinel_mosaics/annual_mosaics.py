from pystac_client import Client
from odc.stac import load
import odc.geo
import xarray as xr
import rioxarray as rio
import geopandas as gpd 
import itertools
# from multiprocessing import Pool
import dask.distributed
import dask.utils
from odc.stac import configure_rio, stac_load

from odc.algo import erase_bad, mask_cleanup

CLOUD_SHADOWS = 3
CLOUD_HIGH_PROBABILITY = 9

bitmask = 0
for field in [CLOUD_SHADOWS, CLOUD_HIGH_PROBABILITY]:
    bitmask |= 1 << field

def process_patch(args: tuple):

    geobox = args[0]
    year = args[1]

    bbox = geobox.boundingbox.to_crs('EPSG:4326')
    bbox = (bbox.left,bbox.bottom,bbox.right,bbox.top)

    green_season = f'{year}-04-01/{year}-10-01'

    catalog = Client.open("https://earth-search.aws.element84.com/v1") 
    search = catalog.search(
        collections = ['sentinel-2-l2a'],
        bbox = bbox, 
        datetime = green_season,
        query = ['eo:cloud_cover<50']
    )

    item_collection = search.item_collection()

    src_dataset = odc.stac.load(
        item_collection,
        bands = ['red', 'green', 'blue', 'nir', 'swir16', 'swir22', 'rededge1', 'rededge2', 'rededge3', 'nir08','scl'],
        geobox = geobox,
        chunks = {'x':2000,'y':2000},
        groupby = 'solar_day',
        resampling = 'bilinear'
    )

    cloud_mask = src_dataset.scl.astype("uint16") & bitmask != 0
    cloud_mask = mask_cleanup(cloud_mask) # Use default filters

    src_dataset = erase_bad(src_dataset, cloud_mask)
    src_dataset = src_dataset.where(src_dataset == 0)
    
    # Calculate the median composite. Maybe put fillna(0) before computing
    target_dataset = src_dataset['red', 'green', 'blue', 'nir', 'swir16', 'swir22', 'rededge1', 'rededge2', 'rededge3', 'nir08'].median(dim='time',skipna=True).fillna(0).astype('uint16').compute()
    # target_dataset = src_dataset['red', 'green', 'blue', 'nir', 'swir16', 'swir22', 'rededge1', 'rededge2', 'rededge3', 'nir08'].median(dim='time',skipna=True).compute().fillna(0).astype('uint16')

    print('Processed median', '\nBounding Box:', bbox, '\nYear:', year)

    target_dataset.rio.to_raster(f'/Users/diegobengochea/git/iberian.carbon/data/Sentinel2_Composites_Spain/sentinel2_mosaic_{year}_lat{bbox[3]}_lon{bbox[0]}_{resolution}m.tif')

if __name__ == '__main__':
    # Start dask cluster
    client = dask.distributed.Client()
    configure_rio(cloud_defaults=True, client = client)

    # Prepare region of interest
    # spain = gpd.read_file('/Users/diegobengochea/git/iberian.carbon/data/SpainPolygon/gadm41_ESP_1.shp')
    spain = gpd.read_file('/home/dibepa/git/iberian.carbon/spain/gadm41_ESP_1.shp')
    # Filter continental Spain
    spain = spain[ (spain.GID_1 != 'ESP.7_1') & (spain.GID_1 != 'ESP.13_1') & (spain.GID_1 != 'ESP.14_1') ] 
    # Reproject to UTM30 EPSG:25830
    spain = spain.dissolve()[['geometry','COUNTRY']].to_crs(epsg='25830')
    # Add CRS information to shapely polygon
    geometry_spain = odc.geo.geom.Geometry(spain.geometry[0],crs='EPSG:25830')
    # Create a GeoBox for all continental Spain with a 10 meters resolution 
    geobox_spain = odc.geo.geobox.GeoBox.from_geopolygon(geometry_spain,resolution=10) # The resolution here is irrelevant since the Spain GeoBOX is too large to make queries, adn thus cannot be used as intersect. Only used to create tiles of suitable shape 20km2

    # Divide the full geobox in Geotiles of smaller size for processing
    geotiles_spain = odc.geo.geobox.GeoboxTiles(geobox_spain,(10000,10000))
    geotiles_spain = [ geotiles_spain.__getitem__(tile) for tile in geotiles_spain._all_tiles() ]

    args = list(itertools.product(geotiles_spain, [2018,2019]))

    print(len(args)*0.5)