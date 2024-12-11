import os
import re
from pathlib import Path
import rasterio
from rasterio.merge import merge
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def group_rasters_by_year(input_dir):
    """
    Group raster files by year based on filename pattern S2_summer_mosaics_YYYY_*.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the downsampled raster files
        
    Returns:
    --------
    dict : Dictionary with years as keys and lists of file paths as values
    """
    pattern = re.compile(r'S2_summer_mosaic_(\d{4})_.*_downsampled\.tif$', re.IGNORECASE)
    raster_groups = defaultdict(list)
    
    for file in os.listdir(input_dir):
        match = pattern.match(file)
        if match:
            year = match.group(1)
            raster_groups[year].append(os.path.join(input_dir, file))
            
    return raster_groups

def merge_rasters_by_year(input_dir, output_dir):
    """
    Merge raster files grouped by year.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the downsampled raster files
    output_dir : str
        Directory where merged rasters will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Group rasters by year
    raster_groups = group_rasters_by_year(input_dir)
    
    if not raster_groups:
        print("No raster files found matching the pattern 'S2_summer_mosaics_YYYY_*_downsampled.tif'")
        return
    
    # Process each year
    for year, raster_files in tqdm(raster_groups.items(), desc="Processing years"):
        output_path = os.path.join(output_dir, f'S2_summer_mosaic_{year}_merged.tif')
        
        try:
            # Open all raster files
            src_files = []
            for raster_path in raster_files:
                src = rasterio.open(raster_path)
                src_files.append(src)
            
            # Merge rasters
            mosaic, out_transform = merge(src_files)
            
            # Get metadata from first raster
            out_meta = src_files[0].meta.copy()
            
            # Update metadata
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "compress": "LZW"  # Add compression to save space
            })
            
            # Write merged raster
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(mosaic)
            
            print(f"\nSuccessfully merged {len(raster_files)} rasters for year {year}")
            
        except Exception as e:
            print(f"\nError processing year {year}: {str(e)}")
            continue
            
        finally:
            # Close all opened raster files
            for src in src_files:
                src.close()
    
    print("\nMerging completed!")

# Example usage
if __name__ == "__main__":
    input_directory = "/Users/diegobengochea/git/iberian.carbon/data/S2_mosaics_downsampled/"  # Directory with downsampled rasters
    output_directory = "/Users/diegobengochea/git/iberian.carbon/data/S2_mosaics_merged/"      # Directory for merged outputs
    merge_rasters_by_year(input_directory, output_directory)

    