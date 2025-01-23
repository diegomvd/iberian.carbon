import os
import re
import rasterio
from rasterio.enums import Resampling
import numpy as np
from pathlib import Path
from tqdm import tqdm

def downsample_rasters(input_dir, output_dir, scale_factor=10):
    """
    Downsample raster files matching pattern 'S2_summer_mosaics_YYYY_*.tif' in the input directory 
    by a given scale factor and save them to the output directory.
    
    Parameters:
    -----------
    input_dir : str
        Path to directory containing input raster files
    output_dir : str
        Path to directory where downsampled files will be saved
    scale_factor : int
        Factor by which to downsample the raster (default: 10)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the pattern to match
    pattern = re.compile(r'S2_summer_mosaic_\d{4}_.*\.tif$', re.IGNORECASE)
    # pattern = re.compile(r'S2_summer_mosaic_\d{4}', re.IGNORECASE)

    
    # Get all files in input directory that match the pattern
    raster_files = [
        f for f in os.listdir(input_dir) 
        if pattern.match(f)
    ]
    
    if not raster_files:
        print("No files found matching the pattern 'S2_summer_mosaics_YYYY_*.tif'")
        return
    
    # Create progress bar
    pbar = tqdm(raster_files, desc="Processing rasters", unit="file")
    
    for raster_file in pbar:
        # Update progress bar description
        pbar.set_description(f"Processing {raster_file}")
        
        input_path = os.path.join(input_dir, raster_file)
        
        # Create output filename
        filename = Path(raster_file).stem
        output_filename = f"{filename}_downsampled.tif"
        output_path = os.path.join(output_dir, output_filename)
        
        if not os.path.exists(output_path):
            try:
                with rasterio.open(input_path) as dataset:
                    # Calculate new dimensions
                    new_height = dataset.height // scale_factor
                    new_width = dataset.width // scale_factor
                    
                    # Calculate new transform
                    transform = dataset.transform * dataset.transform.scale(
                        (dataset.width / new_width),
                        (dataset.height / new_height)
                    )
                    
                    # Create output profile
                    profile = dataset.profile.copy()
                    profile.update({
                        'height': new_height,
                        'width': new_width,
                        'transform': transform
                    })
                    
                    # Read and resample data
                    data = dataset.read(
                        out_shape=(dataset.count, new_height, new_width),
                        resampling=Resampling.average
                    )
                    
                    # Write output file
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(data)
                
            except Exception as e:
                print(f"\nError processing {raster_file}: {str(e)}")
                continue
    
    pbar.close()
    print("\nDownsampling completed!")

# Example usage
if __name__ == "__main__":
    input_directory = "/Users/diegobengochea/git/iberian.carbon/data/S2_PNOA_DATASET/"
    output_directory = "/Users/diegobengochea/git/iberian.carbon/data/S2_mosaics_downsampled/"
    downsample_rasters(input_directory, output_directory)