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
NODATA_OUT = -9999
DTYPE_OUT = 'float32'
PERCENTILE_LOW = 1
PERCENTILE_HIGH = 99

class NormalizationError(Exception):
    """Custom exception for normalization-specific errors."""
    pass

def setup_logger(output_dir: str) -> logging.Logger:
    """Setup logger with both file and console handlers."""
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('raster_normalization')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(
        log_dir / f'normalization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
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
        try:
            if 'client' in locals() and client is not None:
                logger.info("Closing client...")
                client.close()
            if 'cluster' in locals() and cluster is not None:
                logger.info("Closing cluster...")
                cluster.close()
            gc.collect()
            time.sleep(2)
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            logger.info(f"Memory usage after cleanup: {memory_info.rss / 1024 / 1024:.2f} MB")
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {str(cleanup_error)}")

def process_tile_percentiles(tile_path: str) -> Optional[List[float]]:
    """Process a single tile and return its percentiles."""
    src = None
    data = None
    try:
        with rasterio.open(tile_path) as src:
            data = src.read()
            if data.size == 0:
                raise ValueError("Empty raster data")
            
            # Convert to float and handle nodata
            data = data.astype(np.float32)
            data[data == src.nodata] = np.nan
            
            percentiles = []
            for band in range(data.shape[0]):
                band_data = data[band]
                valid_data = band_data[~np.isnan(band_data)]
                
                if valid_data.size == 0:
                    raise ValueError(f"No valid data in band {band}")
                
                # Compute percentiles directly
                p_low = np.percentile(valid_data, PERCENTILE_LOW)
                p_high = np.percentile(valid_data, PERCENTILE_HIGH)
                percentiles.extend([p_low, p_high])
            
            return percentiles
            
    except Exception as e:
        return None
    finally:
        if src is not None:
            src.close()
        if data is not None:
            del data

def compute_global_percentiles_parallel(tile_paths: List[str], 
                                      num_bands: int = 10,
                                      logger: logging.Logger = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute global percentiles for normalization."""
    try:
        df = pd.DataFrame({'tile_path': tile_paths})
        ddf = dd.from_pandas(df, npartitions=len(tile_paths) // 10)
        
        if logger:
            logger.info(f"Computing percentiles for {len(tile_paths)} tiles...")
        
        results = ddf.tile_path.apply(
            process_tile_percentiles,
            meta=('tile_path', 'object')
        ).compute()
        
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            raise NormalizationError("No valid tiles to compute percentiles")
            
        failed_count = len(results) - len(valid_results)
        if failed_count > 0 and logger:
            logger.warning(f"Failed to process {failed_count} tiles during percentile computation")
        
        # Aggregate results
        all_percentiles = np.array(valid_results)
        p_low = np.zeros(num_bands)
        p_high = np.zeros(num_bands)
        
        for band in range(num_bands):
            idx = band * 2
            # Use median of tile percentiles for robustness
            p_low[band] = np.median(all_percentiles[:, idx])
            p_high[band] = np.median(all_percentiles[:, idx + 1])
        
        if logger:
            logger.info("Global percentiles computed successfully")
            for band in range(num_bands):
                logger.info(f"Band {band}: {p_low[band]:.2f} - {p_high[band]:.2f}")
            
        return p_low, p_high
        
    except Exception as e:
        error_msg = f"Failed to compute global percentiles: {str(e)}"
        if logger:
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
        raise NormalizationError(error_msg)

def normalize_tile_parallel(row: pd.Series) -> Dict:
    """Normalize a single tile using robust percentile scaling."""
    input_path = row['input_path']
    output_path = row['output_path']
    p_low = row['p_low']
    p_high = row['p_high']
    
    # Ensure proper type conversion
    if isinstance(p_low, str):
        p_low = np.fromstring(p_low.strip('[]'), sep=' ')
    if isinstance(p_high, str):
        p_high = np.fromstring(p_high.strip('[]'), sep=' ')
    
    p_low = p_low.astype(np.float32)
    p_high = p_high.astype(np.float32)
    
    try:
        with rasterio.open(input_path) as src:
            data = src.read()
            profile = src.profile.copy()
            profile.update({
                'dtype': DTYPE_OUT,
                'nodata': NODATA_OUT,
                'tiled': True,
                'blockxsize': 256,
                'blockysize': 256,
                'compress': 'lzw',
                'bigtiff': True
            })
            
            data = data.astype(np.float32)
            data[data == src.nodata] = np.nan
            
            normalized_data = np.zeros_like(data, dtype=np.float32)
            
            for band in range(data.shape[0]):
                mask = ~np.isnan(data[band])
                if mask.any():
                    # Simple robust scaling to [0,1]
                    normalized_data[band][mask] = (data[band][mask] - p_low[band]) / (p_high[band] - p_low[band])
                    # Clip to [0,1] range
                    #normalized_data[band][mask] = np.clip(normalized_data[band][mask], 0, 1)
                    
                normalized_data[band][~mask] = NODATA_OUT
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(normalized_data)
            
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
    """Process all tiles for a given year."""
    logger = None
    client = None
    
    logger = setup_logger(output_dir)
    logger.info(f"Starting processing for year {year}")
    
    try:
        with setup_dask_cluster(n_workers=n_workers, logger=logger) as client:
            tile_paths = list(Path(input_dir).glob(f'*{year}*.tif'))
            if not tile_paths:
                raise NormalizationError(f"No tiles found for year {year}")
            
            logger.info(f"Found {len(tile_paths)} tiles for year {year}")
            
            # Compute global percentiles
            p_low, p_high = compute_global_percentiles_parallel(tile_paths, logger=logger)
            
            client.restart()
            
            output_paths = [Path(output_dir) / tile_path.name for tile_path in tile_paths]
            
            logger.info("Starting tile normalization...")
            
            df = pd.DataFrame({
                'input_path': [str(p) for p in tile_paths],
                'output_path': [str(p) for p in output_paths],
                'p_low': [p_low.copy() for _ in range(len(tile_paths))],
                'p_high': [p_high.copy() for _ in range(len(tile_paths))]
            })
            
            logger.info(f"DataFrame dtypes: {df.dtypes}")
            
            ddf = dd.from_pandas(df, npartitions=n_workers)
            
            results = ddf.apply(
                normalize_tile_parallel,
                axis=1,
                meta=('results', 'object')
            ).compute()
            
            successful = [r for r in results if r['status'] == 'success']
            failed = [r for r in results if r['status'] == 'error']
            
            logger.info(f"Successfully processed {len(successful)} tiles")
            
            if failed:
                logger.warning(f"Failed to process {len(failed)} tiles")
                for fail in failed:
                    logger.error(f"Failed tile {fail['input_path']}: {fail['error']}")
            
            # Save percentiles
            percentiles_file = Path(output_dir) / f'normalization_percentiles_{year}.npz'
            np.savez(percentiles_file, p_low=p_low, p_high=p_high)
            logger.info(f"Saved normalization percentiles to {percentiles_file}")
            
            client.restart()
            logger.info("Processing completed")
            
            return p_low, p_high
            
    except Exception as e:
        logger.error(f"Critical error during processing: {str(e)}")
        logger.debug(traceback.format_exc())
        raise
    finally:
        if client is not None:
            client.close()

if __name__ == "__main__":
    input_dir = "/Users/diegobengochea/git/iberian.carbon/data/S2_summer_mosaics/"
    output_dir = "/Users/diegobengochea/git/iberian.carbon/data/S2_summer_mosaics_percentile_normalized/"
    years = ["2017","2018","2019","2020","2021"] #["2017","2018","2019","2020","2021","2022","2023","2024"]
    
    try:
        for year in years:
            p_low, p_high = process_year_tiles_parallel(
                input_dir=input_dir,
                output_dir=output_dir,
                year=year,
                n_workers=6
            )
    except Exception as e:
        print(f"Failed to process tiles: {str(e)}")
        sys.exit(1)