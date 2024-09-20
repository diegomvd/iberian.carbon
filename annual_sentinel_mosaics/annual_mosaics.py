from pystac_client import Client
from odc.stac import load
import odc.geo
import xarray as xr
import rioxarray as rio
import geopandas as gpd 
import itertools
from multiprocessing import Pool
import dask.distributed
import dask.utils
from odc.stac import configure_rio, stac_load

def process_patch(args: tuple):

    geobox = args[0]
    year = args[1]

    bbox = geobox.boundingbox.to_crs('EPSG:4326')
    bbox = (bbox.left,bbox.bottom,bbox.right,bbox.top)

    print('Bounding Box:', bbox)
    print('Year:', year)

    green_season = f'{year}-04-01/{year}-10-01'

    catalog = Client.open("https://earth-search.aws.element84.com/v1") 
    search = catalog.search(
        collections = ['sentinel-2-l2a'],
        bbox = bbox, 
        datetime = green_season,
        query = ['eo:cloud_cover<50']
    )

    item_collection = search.item_collection()

    src_dataset_lazy = odc.stac.load(
        item_collection,
        bands = ['red', 'green', 'blue', 'nir', 'swir16', 'swir22', 'rededge1', 'rededge2', 'rededge3', 'nir08'],
        geobox = geobox,
        chunks = {'x':2000,'y':2000}
    )

    src_dataset = src_dataset_lazy.compute()
    print('Loaded data', '\nBounding Box:', bbox, '\nYear:', year)

    # Calculate the median composite
    target_dataset= src_dataset.median(dim='time',skipna=True)
    target_dataset.rio.to_raster(f'test_composite_{year}_lat{bbox[3]}_lon{bbox[0]}_{resolution}m.tif')

if __name__ == '__main__':

    # Start dask cluster
    client = dask.distributed.Client()
    configure_rio(cloud_defaults=True, client = client)

    # Prepare region of interest
    spain = gpd.read_file('/Users/diegobengochea/git/iberian.carbon/data/SpainPolygon/gadm41_ESP_1.shp')
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

    # List of bounding boxes to query the sentinel-2-l2a catalog
   # geotiles_bbox_latlon = [ geotiles_spain.__getitem__(tile).boundingbox.to_crs('EPSG:4326') for tile in geotiles_spain._all_tiles()]
    #geotiles_bbox_latlon = [(bbox.left,bbox.bottom,bbox.right,bbox.top) for bbox in geotiles_bbox_latlon]

    args = list(itertools.product(geotiles_spain, [2018,2019]))
    print(args[0])

    print(len(args))

    #with Pool(20) as pool:
    #    print('Beggining process')
    #    result = pool.map(process_patch, args)