import xarray as xr
import numpy as np
import dask.array as da
from pystac_client import Client
from datetime import datetime

def assess_median_robustness(scenes, min_scenes=5, max_scenes=30, step=5):
    """
    Assess the robustness of median mosaics with different numbers of scenes
    
    Args:
        scenes: List of xarray DataArrays
        min_scenes: Minimum number of scenes to test
        max_scenes: Maximum number of scenes to test
        step: Step size for number of scenes
    """
    total_scenes = len(scenes)
    results = {}
    
    # Stack all scenes
    stacked_scenes = xr.concat(scenes, dim="time")
    
    # Reference mosaic using all scenes
    reference_mosaic = stacked_scenes.median(dim="time")
    
    for n_scenes in range(min_scenes, min(max_scenes, total_scenes), step):
        # Perform multiple random samples
        differences = []
        valid_pixel_counts = []
        
        for _ in range(10):  # 10 random samples for each n_scenes
            # Randomly sample n scenes
            sample_indices = np.random.choice(total_scenes, n_scenes, replace=False)
            sampled_scenes = stacked_scenes.isel(time=sample_indices)
            
            # Create median mosaic
            sample_mosaic = sampled_scenes.median(dim="time")
            
            # Calculate difference from reference
            diff = abs(sample_mosaic - reference_mosaic)
            differences.append(float(diff.mean().values))
            
            # Calculate valid pixel percentage
            valid_pixels = (~sample_mosaic.isnull()).sum() / sample_mosaic.size
            valid_pixel_counts.append(float(valid_pixels))
        
        results[n_scenes] = {
            'mean_difference': np.mean(differences),
            'std_difference': np.std(differences),
            'mean_valid_pixels': np.mean(valid_pixel_counts),
            'std_valid_pixels': np.std(valid_pixel_counts)
        }
    
    return results, reference_mosaic

# Example usage
stac_url = "https://earth-search.aws.element84.com/v0"
client = Client.open(stac_url)

bbox = [-122.34, 37.74, -122.26, 37.80]
start_date = datetime(2022, 1, 1)
end_date = datetime(2022, 12, 31)

search = client.search(
    collections=["sentinel-s2-l2a-cogs"],
    bbox=bbox,
    datetime=f"{start_date.isoformat()}/{end_date.isoformat()}",
)

scenes = []
for item in search.get_items():
    scene = load_and_mask_scene(item)  # Using previous masking function
    scenes.append(scene)

results, reference = assess_median_robustness(scenes)

# Print results
for n_scenes, stats in results.items():
    print(f"\nNumber of scenes: {n_scenes}")
    print(f"Mean difference from reference: {stats['mean_difference']:.4f}")
    print(f"Valid pixel percentage: {stats['mean_valid_pixels']*100:.1f}%")
Last edited 11 hours ago
