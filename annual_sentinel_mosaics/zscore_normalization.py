import numpy as np
import rasterio
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
import logging
from datetime import datetime
import traceback
from typing import List, Tuple, Optional, Dict
import sys
import contextlib
import gc
import psutil
from contextlib import contextmanager
import time

# Constants
NODATA_OUT = -9999  # New nodata value for output
DTYPE_OUT = 'float32'  # Output datatype

class NormalizationError(Exception):
    """Custom exception for normalization-specific errors."""
    pass

def setup_logger(output_dir: str) -> logging.Logger:
    """Setup logger with both file and console handlers."""
    # Create logs directory
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('raster_normalization')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(
        log_dir / f'normalization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

@contextmanager
def setup_dask_cluster(n_workers: Optional[int] = None, 
                      threads_per_worker: Optional[int] = None,
                      logger: logging.Logger = None) -> Client:
    """Setup a local Dask cluster with error handling."""
    try:
        if n_workers is None:
            n_workers = 3
        if threads_per_worker is None:
            threads_per_worker = 3

        try:
            client = Client('localhost:8787')
            client.close()
            del client
        except:
            pass
        
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit='28GB'
        )
        client = Client(cluster)
        
        if logger:
            logger.info(f"Dask cluster initialized with {n_workers} workers")
            logger.info(f"Dashboard available at: {client.dashboard_link}")
        
        yield client
    
    except Exception as e:
        error_msg = f"Failed to setup Dask cluster: {str(e)}"
        if logger:
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
        raise NormalizationError(error_msg)
    finally:
        # Cleanup sequence                                                                                                                                    
        try:
            if client is not None:
                logger.info("Closing client...")
                client.close()

            if cluster is not None:
                logger.info("Closing cluster...")
                cluster.close()

            # Force garbage collection                                                                                                                        
            gc.collect()

            # Wait a moment for resources to be released                                                                                                      
            time.sleep(2)

            # Log memory usage after cleanup                                                                                                                  
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            logger.info(f"Memory usage after cleanup: {memory_info.rss / 1024 / 1024:.2f} MB")

        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {str(cleanup_error)}")

def process_tile_stats(tile_path: str) -> Optional[List[float]]:
    """
    Process a single tile and return its statistics, handling nodata values properly.
    """
    src = None
    data = None
    try:
        with rasterio.open(tile_path) as src:
            data = src.read()
            if data.size == 0:
                raise ValueError("Empty raster data")
            
            # Replace 0 (nodata) with nan
            data = data.astype(np.float32)  # Convert to float to handle nan
            data[data == src.nodata] = np.nan
                
            stats = []
            for band in range(data.shape[0]):
                band_data = data[band]
                valid_data = band_data[~np.isnan(band_data)]
                
                if valid_data.size == 0:
                    raise ValueError(f"No valid data in band {band}")
                    
                stats.extend([
                    np.nansum(band_data),
                    np.nansum(band_data ** 2),
                    np.sum(~np.isnan(band_data))  # Count of non-nan values
                ])
            
            return stats
            
    except Exception as e:
        return None
    finally:
        if src is not None:
            src.close()
        if data is not None:
            del data
    
def compute_global_stats_parallel(tile_paths: List[str], 
                                num_bands: int = 10,
                                logger: logging.Logger = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute global statistics with proper handling of nodata values.
    """
    try:

        df = pd.DataFrame({'tile_path': tile_paths})
        ddf = dd.from_pandas(df, npartitions=len(tile_paths) // 10)
        
        if logger:
            logger.info(f"Computing statistics for {len(tile_paths)} tiles...")
        
        # Specify meta for the apply operation
        results = ddf.tile_path.apply(
            process_tile_stats,
            meta=('tile_path', 'object')  # 'object' type since we return Optional[List[float]]
        ).compute()
        
        # Filter out None results from failed tiles
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            raise NormalizationError("No valid tiles to compute statistics")
            
        failed_count = len(results) - len(valid_results)
        if failed_count > 0 and logger:
            logger.warning(f"Failed to process {failed_count} tiles during statistics computation")
            
        # Aggregate results
        total_stats = np.sum(valid_results, axis=0)
        means = np.zeros(num_bands)
        stds = np.zeros(num_bands)
        
        for band in range(num_bands):
            idx = band * 3
            sum_values = total_stats[idx]
            sum_squares = total_stats[idx + 1]
            count = total_stats[idx + 2]
            
            means[band] = sum_values / count
            stds[band] = np.sqrt(sum_squares/count - means[band]**2)
        
        if logger:
            logger.info("Global statistics computed successfully")
            
        return means, stds
        
    except Exception as e:
        error_msg = f"Failed to compute global statistics: {str(e)}"
        if logger:
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
        raise NormalizationError(error_msg)

def normalize_tile_parallel(row: pd.Series) -> Dict:
    """
    Normalize a single tile with proper nodata handling.
    
    Args:
        row (pd.Series): Row containing input_path, output_path, means, and stds
        
    Returns:
        Dict: Status dictionary containing processing results
    """
    input_path = row['input_path']
    output_path = row['output_path']
    means = row['means']
    stds = row['stds']
    
    # Ensure proper type conversion for means and stds
    if isinstance(means, str):
        means = np.fromstring(means.strip('[]'), sep=' ')
    if isinstance(stds, str):
        stds = np.fromstring(stds.strip('[]'), sep=' ')
    
    # Ensure arrays are float32
    means = means.astype(np.float32)
    stds = stds.astype(np.float32)
    
    try:
        with rasterio.open(input_path) as src:
            data = src.read()
            profile = src.profile.copy()
           # profile.update(dtype=DTYPE_OUT, nodata=NODATA_OUT)
            profile.update({
                'dtype': DTYPE_OUT,
                'nodata': NODATA_OUT,
                'tiled': True,
                'blockxsize': 256,  # Standard tile size
                'blockysize': 256,
                'compress': 'lzw',  # LZW compression
                'bigtiff': True     # Enable BigTIFF support for large files
            })
            # Convert to float32 and handle nodata
            data = data.astype(np.float32)
            data[data == src.nodata] = np.nan
            
            # Preallocate normalized data array
            normalized_data = np.zeros_like(data, dtype=np.float32)
            
            # Normalize each band
            for band in range(data.shape[0]):
                mask = ~np.isnan(data[band])
                normalized_data[band][mask] = (data[band][mask] - means[band]) / stds[band]
                normalized_data[band][~mask] = NODATA_OUT
            
            # Write normalized data
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(normalized_data)
            
            
            # Clean up
            del data
            del normalized_data
            gc.collect()
            
            return {
                'status': 'success',
                'input_path': input_path,
                'output_path': output_path
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'input_path': input_path,
            'error': str(e)
        }

def process_year_tiles_parallel(input_dir: str,
                              output_dir: str,
                              year: str,
                              n_workers: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process all tiles for a given year with nodata handling.
    """

    logger = None
    client = None
    
    # Setup logger
    logger = setup_logger(output_dir)
    logger.info(f"Starting processing for year {year}")
    
    try:
        # Setup Dask cluster
        with setup_dask_cluster(n_workers=n_workers, logger=logger) as client:
        
            # Get all tiles for this year
            tile_paths = list(Path(input_dir).glob(f'*{year}*.tif'))
            if not tile_paths:
                raise NormalizationError(f"No tiles found for year {year}")
            
            logger.info(f"Found {len(tile_paths)} tiles for year {year}")
        
            # Compute global statistics
            means, stds = compute_global_stats_parallel(tile_paths, logger=logger)

            # Clear memory after computing stats
            client.restart()
        
            # Prepare normalization tasks
            output_paths = [Path(output_dir) / tile_path.name for tile_path in tile_paths]

            # Prepare normalization tasks

            logger.info("Starting tile normalization...")
            # Create dataframe with separate columns
            # Create dataframe with explicit dtypes to prevent string conversion
            # Create dataframe ensuring numpy arrays are properly serialized
            # In process_year_tiles_parallel:
            df = pd.DataFrame({
                'input_path': [str(p) for p in tile_paths],
                'output_path': [str(p) for p in output_paths],
                'means': [means.copy() for _ in range(len(tile_paths))],  # Convert to list for proper serialization
                'stds': [stds.copy() for _ in range(len(tile_paths))]
            })


            # Verify dtypes
            logger.info(f"DataFrame dtypes: {df.dtypes}")
        
            # Verify that means and stds are still numpy arrays
            if not isinstance(df['means'].iloc[0], np.ndarray):
                raise TypeError(f"means has been converted to {type(df['means'].iloc[0])}")
            if not isinstance(df['stds'].iloc[0], np.ndarray):
                raise TypeError(f"stds has been converted to {type(df['stds'].iloc[0])}")
        
            ddf = dd.from_pandas(df, npartitions=n_workers)
        
            # Specify meta for the dictionary output
            #results = ddf.map_partitions(
            #    lambda partition: partition.apply(normalize_tile_parallel, axis=1),
            #    meta=('dict', 'object')
            #).compute()

            # Process using apply
            results = ddf.apply(
                normalize_tile_parallel,
                axis=1,
                meta=('results', 'object')
            ).compute()

        
            # Process results
            successful = [r for r in results if r['status'] == 'success']
            failed = [r for r in results if r['status'] == 'error']
        
            logger.info(f"Successfully processed {len(successful)} tiles")
        
            if failed:
                logger.warning(f"Failed to process {len(failed)} tiles")
                for fail in failed:
                    logger.error(f"Failed tile {fail['input_path']}: {fail['error']}")
        
            # Save statistics
            stats_file = Path(output_dir) / f'normalization_stats_{year}.npz'
            np.savez(stats_file, means=means, stds=stds)
            logger.info(f"Saved normalization statistics to {stats_file}")
        
            client.restart()
            logger.info("Processing completed")

        return means, stds
        
    except Exception as e:
        logger.error(f"Critical error during processing: {str(e)}")
        logger.debug(traceback.format_exc())
        raise
    finally:
        if client is not None:
            client.close()

if __name__ == "__main__":
    # Configuration
    input_dir = "/Users/diegobengochea/git/iberian.carbon/data/S2_summer_mosaics/"
    output_dir = "/Users/diegobengochea/git/iberian.carbon/data/S2_summer_mosaics_normalized/"
    years = ["2017","2018","2019","2020","2021","2022","2023","2024"]
    
    try:
        for year in years:
            means, stds = process_year_tiles_parallel(
                input_dir=input_dir,
                output_dir=output_dir,
                year=year,
                n_workers=6
            )
    except Exception as e:
        print(f"Failed to process tiles: {str(e)}")
        sys.exit(1)
