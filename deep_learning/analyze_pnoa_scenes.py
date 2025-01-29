import rasterio
import numpy as np
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

@dataclass
class TileMetrics:
    """Stores computed metrics for a vegetation height tile"""
    file_path: str
    valid_pixels: int
    total_pixels: int
    height_mean: float
    height_median: float
    height_std: float
    height_iqr: float
    height_skewness: float
    shannon_diversity: float
    artificial_ratio: float
    percentile_10: float
    height_distribution: Dict[str, float]
    percentile_90: float
    percentile_95: float

class HeightRasterAnalyzer:
    """Analyzes vegetation height raster files for dataset filtering"""
    
    def __init__(
        self,
        data_dir: str = "/Users/diegobengochea/git/iberian.carbon/data/S2_PNOA_DATASET/",
        nodata_value: float = -32767.0,
        low_veg_threshold: float = 1.0,
        height_bins: List[float] = [0, 1, 2, 4, 8, 12, 16, 20, 25, float('inf')],
        n_workers: int = 22
    ):
        self.data_dir = Path(data_dir)
        self.nodata_value = nodata_value  # This value flags artificial surfaces
        self.low_veg_threshold = low_veg_threshold
        self.height_bins = height_bins
        self.n_workers = n_workers
        self.logger = self._setup_logger()
        
        # Define pruning strategies
        self.strategies = {
            "conservative": {
                "max_artificial_ratio": 0.5,    # Allow more artificial surfaces
                "percentile_10": 0.1,       # Much more permissive with low vegetation
                "percentile_90": 13.0,      # Only require minimal tall vegetation
                "min_diversity": 0.0             # Lower diversity requirement
            },
            "moderate": {
                "max_artificial_ratio": 0.5,    # Moderate restriction on artificial
                "percentile_10": 0.1,       # Much more permissive with low vegetation
                "percentile_90": 15.0,      # Only require minimal tall vegetation
                "min_diversity": 0.0             # Medium diversity requirement
            },
            "aggressive": {
                "max_artificial_ratio": 0.5,    # Strict on artificial surfaces
                "percentile_10": 0.1,       # Much more permissive with low vegetation
                "percentile_90": 17.0,      # Only require minimal tall vegetation
                "min_diversity": 0.0             # High diversity requirement
            }
        }
        
    def _setup_logger(self) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
        return logger

    def _compute_shannon_diversity(self, heights: np.ndarray) -> float:
        """
        Compute Shannon diversity index for height distribution
        Higher values indicate more diverse height ranges
        """
        hist, _ = np.histogram(heights, bins=self.height_bins)
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist))

    def _analyze_single_tile(self, file_path: Path) -> TileMetrics:
        """Analyze a single raster tile and compute metrics"""
        try:
            with rasterio.open(file_path) as src:

                # self.logger.info(f"Processing {file_path}")                

                data = src.read(1)
                
                # Create masks
                artificial_mask = (data == self.nodata_value)

                wrong_data_mask = (data > 60.0)

                valid_mask = (~artificial_mask) & (~wrong_data_mask)

                if not valid_mask.any():
                    return None
                
                valid_heights = data[valid_mask]
                
                # self.logger.info(f"Calculating height distribution")     
                # Calculate height distribution
                hist, _ = np.histogram(valid_heights, bins=self.height_bins)
                height_dist = {f"{self.height_bins[i]:.1f}-{self.height_bins[i+1]:.1f}m": 
                             hist[i]/len(valid_heights) for i in range(len(hist))}
                

                # self.logger.info(f"Calculating metrics")   
                metrics = TileMetrics(
                    file_path=str(file_path),
                    valid_pixels=valid_mask.sum(),
                    total_pixels=data.size,
                    height_mean=np.mean(valid_heights),
                    height_median=np.median(valid_heights),
                    height_std=np.std(valid_heights),
                    height_iqr=np.percentile(valid_heights, 75) - np.percentile(valid_heights, 25),
                    height_skewness=float(pd.Series(valid_heights).skew()),
                    shannon_diversity=self._compute_shannon_diversity(valid_heights),
                    artificial_ratio=1.0-valid_mask.sum()/data.size,  #artificial_mask.sum() / data.size,
                    percentile_10=np.percentile(valid_heights, 10),
                    height_distribution=height_dist,
                    percentile_90=np.percentile(valid_heights, 90),
                    percentile_95=np.percentile(valid_heights, 95)
                )
                return metrics
                
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            return None

    def analyze_pruning_strategies(
        self,
        metrics_df: pd.DataFrame,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, float]]]:
        """
        Analyze three pruning strategies with increasing aggressiveness.
        Returns filtered DataFrames and statistics for each strategy.
        """
        filtered_dfs = {}
        stats = {}
        total_pixels = metrics_df['valid_pixels'].sum()
        
        # print(metrics_df)
        # print(metrics_df['height_mean'])
        print(metrics_df['artificial_ratio'].max())
        print(metrics_df['artificial_ratio'].min())
        # print(metrics_df['height_mean'].unique())

        for name, thresholds in self.strategies.items():
            # Apply filters
            mask = (
                (metrics_df['artificial_ratio'] <= thresholds['max_artificial_ratio']) &
                (metrics_df['percentile_10'] >= thresholds['percentile_10']) &
                (metrics_df['percentile_90'] >= thresholds['percentile_90']) #&  # Ensures presence of tall vegetation
                # (metrics_df['shannon_diversity'] >= thresholds['min_diversity'])
            )
            
            filtered_df = metrics_df[mask].copy()
            
            # Compute statistics
            retained_pixels = filtered_df['valid_pixels'].sum()
            retained_tiles = len(filtered_df)
            
            # Calculate average height distribution
            avg_height_dist = {}
            height_dists = filtered_df['height_distribution'].tolist()
            if height_dists:
                # Get all possible height ranges
                height_ranges = set().union(*[d.keys() for d in height_dists])
                for height_range in height_ranges:
                    values = [d.get(height_range, 0.0) for d in height_dists]
                    avg_height_dist[height_range] = sum(values) / len(values)


            stats[name] = {
                "retained_data_ratio": retained_pixels / total_pixels,
                "retained_tiles_ratio": retained_tiles / len(metrics_df),
                "mean_diversity": filtered_df['shannon_diversity'].mean(),
                "mean_artificial_ratio": filtered_df['artificial_ratio'].mean(),
                "percentile_10": filtered_df['percentile_10'].mean(),
                "percentile_90": filtered_df['percentile_90'].mean(),
                "mean_height": filtered_df['height_mean'].mean(),
                "height_distribution":  avg_height_dist
            }
            
            filtered_dfs[name] = filtered_df
            
        return filtered_dfs, stats

    def plot_pruning_results(
        self,
        original_df: pd.DataFrame,
        strategy_results: Dict[str, pd.DataFrame],
        strategy_stats: Dict[str, Dict[str, float]]
    ) -> None:
        """Plot comparative analysis of pruning strategies using seaborn"""
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 2, height_ratios=[2, 1.5, 1.5])
        
        # Plot 1: Height distributions comparison
        ax1 = fig.add_subplot(gs[0, :])
        height_ranges = list(original_df['height_distribution'].iloc[0].keys())
        
         # Prepare data for plotting
        plot_data = []
        for strategy_name, df in {**{'Original': original_df}, **strategy_results}.items():
            # Calculate average height distribution manually
            height_dists = df['height_distribution'].tolist()
            if height_dists:
                # Get height ranges from first dictionary
                height_ranges = height_dists[0].keys()
                # Calculate mean for each range
                for height_range in height_ranges:
                    values = [d[height_range] for d in height_dists]
                    avg_value = sum(values) / len(values)
                    plot_data.append({
                        'Height Range': height_range,
                        'Proportion': avg_value,
                        'Strategy': strategy_name
                    })
        
        plot_df = pd.DataFrame(plot_data)
        sns.barplot(
            data=plot_df,
            x='Height Range',
            y='Proportion',
            hue='Strategy',
            ax=ax1
        )
        ax1.set_yscale('log')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        ax1.set_title('Height Distribution Comparison (Log Scale)', pad=20)
        
        # Plot 2: Strategy metrics comparison
        ax2 = fig.add_subplot(gs[1, :])
        metrics_data = []
        for strategy, stats in strategy_stats.items():
            for metric, value in stats.items():
                if metric != 'height_distribution':
                    metrics_data.append({
                        'Metric': metric.replace('_', ' ').title(),
                        'Value': value,
                        'Strategy': strategy
                    })
        
        metrics_df = pd.DataFrame(metrics_data)
        sns.barplot(
            data=metrics_df,
            x='Metric',
            y='Value',
            hue='Strategy',
            ax=ax2
        )
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Plot 3: Distribution of key metrics
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[2, 1])
        
        # Shannon diversity distribution
        diversity_data = []
        for name, df in strategy_results.items():
            for val in df['shannon_diversity']:
                diversity_data.append({
                    'Strategy': name,
                    'Shannon Diversity': val
                })
        
        sns.violinplot(
            data=pd.DataFrame(diversity_data),
            x='Strategy',
            y='Shannon Diversity',
            ax=ax3
        )
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        
        # Height distribution
        height_data = []
        for name, df in strategy_results.items():
            for val in df['height_mean']:
                height_data.append({
                    'Strategy': name,
                    'Mean Height (m)': val
                })
        
        sns.violinplot(
            data=pd.DataFrame(height_data),
            x='Strategy',
            y='Mean Height (m)',
            ax=ax4
        )
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()

    def analyze_dataset(self) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, Dict]]:
        """
        Analyze all PNOA raster files and generate strategy-based results
        Returns:
        - DataFrame with metrics
        - Dictionary with files to keep for each strategy
        - Dictionary with strategy statistics
        """
        # Find all PNOA raster files
        raster_files = list(self.data_dir.glob("PNOA_*.tif"))
        self.logger.info(f"Found {len(raster_files)} PNOA raster files")
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:

            results = list(tqdm(
                executor.map(self._analyze_single_tile, raster_files),
                total=len(raster_files),
                desc="Analyzing raster files",
                unit="file"
            ))
        
        self.logger.info(f"Finished raster processing. Summarizing metrics.")   
        # Filter out None results and convert to DataFrame
        valid_results = [r for r in results if r is not None]
        metrics_df = pd.DataFrame([vars(m) for m in valid_results])
        
        self.logger.info(f"Analyzing pruning strategies.")
        # Apply pruning strategies
        strategy_dfs, strategy_stats = self.analyze_pruning_strategies(metrics_df)
        
        # Create file lists for each strategy
        strategy_files = {
            name: df['file_path'].tolist() 
            for name, df in strategy_dfs.items()
        }
        
        self.logger.info(f"Done.")
        return metrics_df, strategy_files, strategy_stats

    def save_results(
        self,
        metrics_df: pd.DataFrame,
        strategy_files: Dict[str, List[str]],
        strategy_stats: Dict[str, Dict],
        output_dir: str = "raster_analysis_results"
    ) -> None:
        """Save analysis results and strategy-based file lists"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save metrics
        metrics_df.to_csv(output_path / "raster_metrics.csv", index=False)
        
        # Save strategy results
        for strategy_name, files in strategy_files.items():
            with open(output_path / f"{strategy_name}_files.txt", 'w') as f:
                f.write("\n".join(files))
        
        # Save strategy statistics
        with open(output_path / "strategy_stats.txt", 'w') as f:
            for strategy, stats in strategy_stats.items():
                f.write(f"\n{strategy.upper()} Strategy:\n")
                f.write("-" * 50 + "\n")
                for metric, value in stats.items():
                    if metric != 'height_distribution':
                        f.write(f"{metric}: {value:.3f}\n")
        
        self.logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    analyzer = HeightRasterAnalyzer()
    metrics_df, strategy_files, strategy_stats = analyzer.analyze_dataset()
    
    # Plot results
    strategy_dfs = {
        name: metrics_df[metrics_df['file_path'].isin(files)]
        for name, files in strategy_files.items()
    }
   
    analyzer.plot_pruning_results(metrics_df, strategy_dfs, strategy_stats)
    
    
    # Save results
    analyzer.save_results(metrics_df, strategy_files, strategy_stats)