import os
import re
import numpy as np
import rasterio
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

def load_raster_data(filepath):
    """
    Load raster data and return as a dictionary of band arrays
    """
    with rasterio.open(filepath) as src:
        return {f"band_{i+1}": src.read(i+1) for i in range(src.count)}

def calculate_statistics(data_array):
    """
    Calculate basic statistics for a single band
    """
    return {
        'mean': np.mean(data_array),
        'std': np.std(data_array),
        'median': np.median(data_array),
        'q25': np.percentile(data_array, 25),
        'q75': np.percentile(data_array, 75),
        'min': np.min(data_array),
        'max': np.max(data_array)
    }

def perform_ks_test(data1, data2):
    """
    Perform Kolmogorov-Smirnov test on samples from two distributions
    """
    # Sample data to make computation feasible
    sample_size = 10000
    sample1 = np.random.choice(data1.flatten(), sample_size)
    sample2 = np.random.choice(data2.flatten(), sample_size)
    
    statistic, p_value = stats.ks_2samp(sample1, sample2)
    return statistic, p_value

def analyze_band_distributions(input_dir, output_dir):
    """
    Analyze band value distributions across years
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the merged raster files
    output_dir : str
        Directory where analysis outputs will be saved
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all merged raster files
    pattern = re.compile(r'S2_summer_mosaic_(\d{4})_merged\.tif$', re.IGNORECASE)
    raster_files = {
        pattern.match(f).group(1): os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if pattern.match(f)
    }
    
    if not raster_files:
        print("No merged raster files found")
        return
    
    # Initialize results storage
    years = sorted(raster_files.keys())
    stats_data = []
    
    # Process each raster
    for year in tqdm(years, desc="Processing years"):
        data = load_raster_data(raster_files[year])
        
        # Calculate statistics for each band
        for band_name, band_data in data.items():
            stats = calculate_statistics(band_data)
            stats['year'] = year
            stats['band'] = band_name
            stats_data.append(stats)
    
    # Convert to DataFrame
    df_stats = pd.DataFrame(stats_data)
    
    # Create visualizations
    
    # 1. Box plots for each band across years
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df_stats, x='band', y='mean', hue='year')
    plt.title('Band Mean Values Distribution by Year')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'band_means_boxplot.png'))
    plt.close()
    
    # 2. Violin plots for distribution comparison
    plt.figure(figsize=(15, 10))
    sns.violinplot(data=df_stats, x='band', y='mean', hue='year')
    plt.title('Band Mean Values Distribution Density by Year')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'band_means_violin.png'))
    plt.close()
    
    # Perform statistical tests between consecutive years
    statistical_tests = []
    
    for band_num in range(1, 11):  # For each band
        band_name = f"band_{band_num}"
        print(f"\nAnalyzing {band_name}:")
        
        for i in range(len(years)-1):
            year1, year2 = years[i], years[i+1]
            
            # Load data for both years
            data1 = load_raster_data(raster_files[year1])[band_name]
            data2 = load_raster_data(raster_files[year2])[band_name]
            
            # Perform KS test
            statistic, p_value = perform_ks_test(data1, data2)
            
            statistical_tests.append({
                'band': band_name,
                'year1': year1,
                'year2': year2,
                'ks_statistic': statistic,
                'p_value': p_value
            })
    
    # Save statistical results
    df_tests = pd.DataFrame(statistical_tests)
    df_tests.to_csv(os.path.join(output_dir, 'statistical_tests.csv'), index=False)
    
    # Save summary statistics
    df_stats.to_csv(os.path.join(output_dir, 'band_statistics.csv'), index=False)
    
    # Create summary report
    with open(os.path.join(output_dir, 'analysis_report.txt'), 'w') as f:
        f.write("Band Distribution Analysis Report\n")
        f.write("===============================\n\n")
        
        for band in df_stats['band'].unique():
            f.write(f"\n{band.upper()} Analysis:\n")
            band_data = df_stats[df_stats['band'] == band]
            
            f.write("\nYearly Means:\n")
            for _, row in band_data.iterrows():
                f.write(f"{row['year']}: {row['mean']:.2f}\n")
            
            f.write("\nYearly Medians:\n")
            for _, row in band_data.iterrows():
                f.write(f"{row['year']}: {row['median']:.2f}\n")
                
            test_results = df_tests[df_tests['band'] == band]
            f.write("\nKS Test Results (between consecutive years):\n")
            for _, row in test_results.iterrows():
                f.write(f"{row['year1']} vs {row['year2']}: ")
                f.write(f"statistic={row['ks_statistic']:.3f}, ")
                f.write(f"p-value={row['p_value']:.3e}\n")
            
            f.write("\n" + "="*50 + "\n")

if __name__ == "__main__":
    input_directory = "/Users/diegobengochea/git/iberian.carbon/data/S2_mosaics_merged/"      # Directory with merged rasters
    output_directory = "/Users/diegobengochea/git/iberian.carbon/data/S2_mosaics_merged/inter_annual_consistency_results/"  # Directory for analysis outputs
    analyze_band_distributions(input_directory, output_directory)