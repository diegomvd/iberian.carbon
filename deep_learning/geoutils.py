"""
This module contains utility functions to handle geospatial data.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
import rasterio.features
import rasterio.windows
import rioxarray as riox
from rioxarray.merge import merge_arrays
from natsort import natsorted
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_image
from shapely.geometry import box
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject



"""
Series of functions to build point dataset by adding features based on geographic location.
"""
def add_feature_from_raster(data, feature_name, feature_data, dtype):
    """
    This function adds a new feature to a dataset by intercepting coordinates of points in the dataset 
    with the tiles of the raster describing the new feature.
    :param data: The current Point dataset in GeoDataFrame format. 
    :param feature_name: The name of the feature to be added.
    :param feature_data: The path to the feature data.
    :param dtype: The data type of the new feature.
    :return: A GeoDataFrame incorporating the new feature. 
    """

    new_data = data.copy()

    # Extract coordinates from the geometry column of the dataset.
    coordinates = [(x,y) for x,y in zip(new_data['geometry'].x , new_data['geometry'].y)]

    # Open the new feature's raster and add the new feature to the dataset by sampling the coordinates.
    raster = rasterio.open(feature_data)
    new_data[feature_name] = [x for x in raster.sample(coordinates)]

    # Aesthetic changes.
    df = pd.DataFrame(new_data).explode( [feature_name] )

    if dtype == None:
        ret = gpd.GeoDataFrame(df).reset_index(drop=True)
    else: 
        ret = gpd.GeoDataFrame(df).reset_index(drop=True).astype({feature_name:dtype}) 

    return ret

def add_feature_from_polygon_layer(data, feature_name, feature_data, col_name):
    """
    This function adds a new feature to a Point dataset by performing a spatial join with a Polygon layer describing the new feature.
    :param data: The current Point dataset in GeoDataFrame format.
    :param feature_name: The name of the feature to be added in the polygon layer.
    :param feature_data: The path to the new feature's polygon layer.
    :param col_name: The name of the new feature in the dataset. 
    :return: A GeoDataFrame incorporating the new feature. 
    TODO: The polygon layer may have tons of unuseful information that we may want to discard.
    """

    if col_name == None:
        col_name = feature_name

    # Project the dataset's geometry to 4326.
    data["geometry"] = data.geometry.to_crs("EPSG:4326")
    
    # Load the polygon layer containing the data on the new feature.
    polygons = gpd.read_file(feature_data)
    polygons = polygons[[feature_name,"geometry"]]
    polygons = polygons.rename({feature_name:col_name},axis="columns")

    # Perform a spatial join between the dataset and the polygon layer.
    return gpd.sjoin(data, polygons).reset_index(drop=True).drop("index_right",axis="columns")

def add_feature(data, feature_name, feature_datafile, col_name = None, dtype = None):
    """
    This function adds a feature to a point dataset either from a raster file or a polygon layer.
    :param data: The current Point dataset in GeoDataFrame format.
    :param feature_name: The name of the feature to be added.
    :param feature_datafile: The path to the new feature's data.
    :param col_name: The name of the new feature in the dataset. Only used with polygon layers, with raster layers this is feature_name. 
    :param dtype: The data type of the new feature.
    :return: A GeoDataFrame incorporating the new feature. 
    """

    fextension = Path(feature_datafile).suffix

    match fextension:
        case ".tiff":
            return add_feature_from_raster(data,feature_name,feature_datafile,dtype)
        case ".tif":
            return add_feature_from_raster(data,feature_name,feature_datafile,dtype)
        case ".shp":
            return add_feature_from_polygon_layer(data,feature_name,feature_datafile,col_name)

def add_features(data, feature_dict):
    """
    This function adds multiple features, from rasters or polygon layers, to a point dataset. Features are stored in a dictionary that stores their name, their corresponding data files, and data types. Refer to function add_feature for parameters to pass in the dictionary.
    :param data: The base point dataset in GeoDataFrame format.
    :param feature_dict: The dictionary containing the features' information.
    :return: A GeoDataFrame incorporating the new features. 
    """
    data_new = data.copy()
    for predictor in feature_dict:
        path, dtype = feature_dict[predictor]
        data_new  = add_feature(data_new, predictor, path, dtype)
    return data_new    


"""
Series of functions to create rasters and windows, merge tiled rasters and perform arithmetic operations with rasters.
"""

def raster_from_array(array, transform, dtype, nodata, filename):
    """
    Create a raster from a data array.
    :param array: The NumPy data array.
    :param transform: A transform object to map the array's rows and columns to geographical coordinates.
    :param dtype: Data type in the resulting raster.
    :param nodata: No-Data value in the resulting raster.
    :param filename: Path to the created raster.
    """
    
    with rasterio.open(
        filename,
        mode="w",
        driver="GTiff",
        height=array.shape[-2],
        width=array.shape[-1],
        count=1,
        dtype=dtype,
        crs="+proj=latlong",
        transform=transform,
        nodata=nodata,
        compress='lzw'
        # **profile
    ) as new_dataset:
        new_dataset.write(array, 1) 

    return filename

def create_windows(path_reference_raster, n_divisions):
    """
    Create a set of windows to process rasters in chunks. Windows are based on a reference raster and a number of divisions to make on each raster dimension.
    :param path_reference_raster: Path to the reference raster.
    :param n_divisions: Number of divisions in each dimension. The total number of windows in n_divisions^2.
    :return: the list of Window objects.  
    """

    with rasterio.open(path_reference_raster) as reference:
    
        # Define window dimensions to process by chunk: processing in n*n windows.
        ww = reference.width/n_divisions
        wh = reference.height/n_divisions

        col_off_array = np.arange(0,reference.width,ww)
        row_off_array = np.arange(0,reference.height,wh)

        window_list = []
        for col_off in col_off_array:
            for row_off in row_off_array:        
                # Create window:
                window = rasterio.windows.Window(col_off,row_off,ww,wh)
                window_list.append((window,col_off,row_off)) 

    return window_list   

def sum_raster_tiles(raster_files, window, col_off, row_off):
    """
    Sum windows from multiple rasters.
    :param raster_files: The list of raster files to sum.
    :param window: The window in which rasters need to be summed.
    :param col_off: Column offset of the window. TODO: This might not be needed as the information is stored in the Window object.
    :param row_off: Row offset of the window. TODO: This might not be needed as the information is stored in the Window object.
    :return: (1) the array storing the resulting sum, (2) the transform object for the window, (3) the latitude of the window upper-left corner, (4) the longitude of the window upper-left corner.  
    """

    sum = []

    for filename in raster_files:
        
        with rasterio.open(filename) as src:

            transform_src = src.transform
            (lon, lat) = transform_src * (col_off,row_off)
            transform_win = rasterio.windows.transform(window, transform_src)

            try: 
                data = src.read(1, window = window)
            except:
                data = src.read(1)    

            if len(sum) == 0:
                sum = data
            else:
                sum += data

    return sum, transform_win, lon, lat   

def sum_rasters(raster_files, window_list, savedir, file_prefix, dtype, nodata):
    """
    Sum multiple rasters processing by window and save the results. 
    :param raster_files: List of paths to the rasters to be summed.
    :param window_list: List of windows to process the rasters by chunks. 
    :param savedir: Directory to save reuslting rasters.
    :param file_prefix: Semantic identifier for the resulting raster files. 
    :param dtype: Data type of resulting raster.
    :param nodata: No-data value of resulting raster.
    :return: List of paths to the files storing the sums for each window.
    """

    out_paths = []
    for window, col_off, row_off in window_list:
    
        sum, transform, lon, lat = sum_raster_tiles(raster_files,window,col_off,row_off)

        fname = file_prefix + "_lon_{}_lat_{}.tif".format(int(lon),int(lat))
        save_path = savedir + fname

        out_path = raster_from_array(sum,transform,dtype,nodata,save_path)
        out_paths.append(out_path)

    return out_paths

def merge_rasters(raster_paths, merged_path):
    """
    Merge multiple raster windows in a single raster file. 
    :param raster_paths: List of paths to the raster files to merge together.
    :param merged_path: Path to the merged raster file.
    :return: The path to the merged raster file.
    """
    
    # Sort paths to ensure memory efficient merge by merging nearby windows first
    sorted_paths = natsorted(raster_paths, key=str)

    raster_list = [ riox.open_rasterio(file) for file in sorted_paths ] 

    aux_list=[]
    
    # Merge rasters by pairs until there's only one.
    while len(raster_list)>1:
        # Discard left-out raster to make sure there is an even number to merge. Normally, this should be entered in the first iteration if ever.
        if not (len(raster_list)%2 == 0):
            aux_list.append(raster_list[-1])
            raster_list = raster_list[0:-1]

        raster_list = [ merge_arrays([tup[0],tup[1]]) for tup in zip(raster_list[0::2], raster_list[1::2]) ]
        print("Remaining rasters {}".format(len(raster_list)))

    # Merge left-out raster with the result of the progressive merges.
    if len(aux_list)>0:
        print("merging aux")
        raster_aux = merge_arrays(aux_list)   

        print("merging final")
        raster_final = merge_arrays([raster_aux,raster_list[0]])

        raster_final.rio.to_raster(merged_path)
    else :
        if len(raster_list)>1:
            print("Error")
        else:    
            raster_list[0].rio.to_raster(merged_path)  

    return merged_path

def rasterize_point_layer(points_df: pd.DataFrame, observable : str, lat_col="lat", lon_col="lon", resolution_degrees=0.00833, nodata = 9999):
    """
    Rasterizes a point dataset. 
    :param points_df: Pandas DataFrame storing the point dataset.
    :param observable: Feature of the point layer to rasterize. Must be the name of the column in the DataFrame.
    :param lat_col: Name of the column storing the latitude data. 
    :param lon:col: Name of the column storing the longitude data. 
    :param resolution_degrees: Resolution of the resulting raster in degrees. 
    :param nodata: No-data value in the resulting raster.
    :return: The raster stored as an xarray object.
    """
        
    points = gpd.GeoDataFrame(
            points_df,
            geometry = gpd.points_from_xy(x=points_df[lon_col], y=points_df[lat_col])
        )
    
    points = points.drop([lon_col,lat_col], axis="columns")

    raster = make_geocube(
                vector_data=points,
                measurements=[observable],
                resolution=(-resolution_degrees, resolution_degrees),
                rasterize_function=rasterize_image, 
                fill = nodata
            )

    return raster

def change_raster_extent(reference, target, savefile):
    """
    Changes the extent of a target raster to match the extent of a reference raster. 
    :param reference: Path to the reference raster.
    :param target: Path to the target raster.
    :param savefile: Path to the resulting raster.
    :return: The path to the resulting raster.
    """

    with rasterio.open(reference) as ref:
        h = ref.shape[-2]
        w = ref.shape[-1]
        geom_bbox = [box(*ref.bounds)]

        with rasterio.open(target) as tar:
            geom_window = rasterio.features.geometry_window(tar,geom_bbox,boundless=True)
            col_off = geom_window.col_off
            row_off = geom_window.row_off
            
            window = rasterio.windows.Window(col_off,row_off,w,h)

            data = tar.read(1, window = window, boundless = True, fill_value = tar.nodata)

            savepath = Path(savefile)
            dir = savepath.parents[0]
            if not dir.exists():
                dir.mkdir()

            with rasterio.open(
                savefile, 
                mode="w",
                driver="GTiff",
                height=h,
                width=w,
                count=1,
                dtype=data.dtype,
                crs="+proj=latlong",
                transform=ref.transform,
                nodata=tar.nodata,
                compress='lzw'
                # **profile
                ) as new_dataset:
                    new_dataset.write(data[:,:], 1)
    return savefile                

def downsample_raster(target_resolution: float, base_raster_file: str, target_raster_file: str):
    """
    Rescales a raster file at a coarser resolution.
    :param target_resolution: The final resolution for the target raster.
    :param base_raster_file: The path to the base raster.
    :param target_raster_file: The saving path to the target raster.
    :return: The path to the newly created target raster at a broader resolution.
    """

    with rasterio.open(base_raster_file) as src:

        width_pixels = src.width 
        bounds = src.bounds
        width_degrees = bounds.right - bounds.left

        # This is the pixel size
        current_resolution = width_degrees / width_pixels 

        if (current_resolution > target_resolution ):
            print("Target pixel size is smaller than origin pixel size. Downsampling aborted.")
        
        else:
            downscale_factor =  current_resolution / target_resolution 

            # Resample data to target shape
            data = src.read(
                1,
                out_shape=(
                    src.count,
                    int(src.height * downscale_factor),
                    int(src.width * downscale_factor)
                ),
                resampling=Resampling.average
            )

            # Scale image transform
            transform = src.transform * src.transform.scale(
                (src.width / data.shape[-1]),
                (src.height / data.shape[-2])
            )

            with rasterio.open(
                target_raster_file,
                mode="w",
                driver="GTiff",
                height=data.shape[-2],
                width=data.shape[-1],
                count=1,
                dtype=data.dtype,
                crs=src.crs,
                transform=transform,
                nodata=src.nodata,
                compress='lzw'
                # **profile
            ) as new_dataset:
                new_dataset.write(data[:,:], 1)
    
    return target_raster_file    

def reproject_raster(target_projection_crs: str, base_raster_file: str, target_raster_file: str):
    """
    Reprojects a raster to a different CRS.
    :param target_projection_crs: The desired CRS for the output raster.
    :param base_raster_file: The path to the base raster.
    :param target_raster_file: The saving path to the target raster.
    :return: The path to the created raster file using the new CRS.
    """

    with rasterio.open(base_raster_file) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_projection_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_projection_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(target_raster_file, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_projection_crs,
                    compress='lzw',
                    resampling=Resampling.average
                    # resampling=Resampling.nearest
                    )

    return target_raster_file        

def rescale_raster(scaling_factor: str, base_raster_file: str, target_raster_file: str):
    """
    Rescale values of a raster.
    :param scaling_factor: The scaling factor.
    :param base_raster_file: The path to the base raster.
    :param target_raster_file: The saving path to the target raster.
    :return: The path to the created raster file with re-scaled values.
    """

    with rasterio.open(base_raster_file) as src:
        
        data = src.read(1) * scaling_factor

        with rasterio.open(
                target_raster_file,
                mode="w",
                driver="GTiff",
                height=data.shape[-2],
                width=data.shape[-1],
                count=1,
                dtype=data.dtype,
                crs=src.crs,
                transform=src.transform,
                nodata=-32768,
                compress='lzw'
                # **profile
            ) as new_dataset:
                new_dataset.write(data[:,:], 1)

    return target_raster_file

def vectorize_raster_layer_to_points(rasterfile,column_name):
    
    with rasterio.open(rasterfile) as src:
    
        image = src.read(1)

        r_indices, c_indices = np.where(image != src.nodata)
        data_indices = list(zip(r_indices,c_indices))
        coordinates = [{'lon': coord[0], 'lat': coord[1] } for coord in [src.xy(row,col) for row,col in data_indices]]

        entries = coordinates.copy()
        data = [{ column_name : image[row,col]} for row,col in data_indices]
        entries = [ d[0]|d[1] for d in zip(data,entries) ]
        df = pd.DataFrame.from_dict(entries)
        gdf = gpd.GeoDataFrame(
            df,
            geometry= gpd.points_from_xy(x=df.lon, y=df.lat)
        )
        gdf.crs = "EPSG:4326"

    return gdf

"""
Distance functions for clustering and autocorrelation estimation purposes.
"""

def haversine_distance_km_from_lat_lon(coord1,coord2):
    """
    Calculates the haversine distance between two points in kilometers from pairs of latitude and longitude. 
    :param coord1: Latitude, longitude tuple (lat,lon) of coordinate 1.
    :param coord2: Latitude, longitude tuple (lat,lon) of coordinate 2.
    :return: The haversine distance in kilometers.
    """
    
    earth_radius = 6371.0088

    lat1 = np.radians(coord1[0])
    lon1 = np.radians(coord1[1])
    lat2 = np.radians(coord2[0])
    lon2 = np.radians(coord2[1])
   
    dlon = lon2 - lon1
    dlat = lat2 - lat1
 
    dist = 2 * np.arcsin(  np.sqrt( (np.sin(dlat/2))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2.0))**2 )   )
 
    dist_km = earth_radius * dist
    return dist_km

def get_radian_coordinates(coordinates: pd.DataFrame, lat_col:str, lon_col:str):
    """
    Transform a pandas DataFrame storing coordinates from degrees to radians. 
    :param coordinates: Degree coordinates DataFrame.
    :param lat_col: Name of the column storing the latitudes.
    :param lon_col: Name of the column storing the longitudes.
    :return: An identical pandas DataFrame with coordinates in radians.
    """

    radian_coordinates = coordinates.copy()

    # Transfoorm coordinates to radians
    radian_coordinates[lat_col] = radian_coordinates[lat_col].apply(lambda lat: np.radians(lat))
    radian_coordinates[lon_col] = radian_coordinates[lon_col].apply(lambda lon: np.radians(lon))

    return radian_coordinates   
