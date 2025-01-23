import os
from collections import defaultdict
import re

def analyze_missing_files(directory_path):
    """
    Analyzes a directory of raster files to identify which location files are missing in specific years
    and generates hypothetical paths for missing files.
    
    Parameters:
    directory_path (str): Path to the directory containing the raster files
    
    Returns:
    tuple: (missing_files dict, all_years list, missing_file_paths list)
    """
    # Get all files in the directory
    files = os.listdir(directory_path)
    
    # Extract years and create a mapping of locations to years
    year_pattern = r'S2_summer_mosaic_(\d{4})'  # Pattern to match 4-digit years
    location_to_years = defaultdict(set)
    all_years = set()
    
    # Dictionary to store the full filenames for each location and year
    location_files = defaultdict(dict)
    
    for file in files:
        # Find the year in the filename
        year_match = re.search(year_pattern, file)
        if year_match:
            year = year_match.group(1)
            all_years.add(year)
            
            # Remove the year from the filename to get the location identifier
            location = re.sub(year_pattern, '', file)
            location_to_years[location].add(year)
            location_files[location][year] = file

    # Find missing years for each location and generate missing file paths
    missing_files = {}
    missing_file_paths = []
    
    for location, years in location_to_years.items():
        missing_years = all_years - years
        if missing_years:
            # Get an example existing file for this location to use as a template
            example_year = next(iter(years))
            example_filename = location_files[location][example_year]
            
            # Store missing years info
            missing_files[location] = {
                'missing_years': sorted(list(missing_years)),
                'existing_files': {
                    year: location_files[location][year]
                    for year in years
                }
            }
            
            # Generate hypothetical paths for missing files
            for missing_year in missing_years:
                # Replace the example year with the missing year in the filename
                missing_filename = re.sub(
                    example_year,
                    missing_year,
                    example_filename
                )
                missing_file_paths.append(os.path.join(directory_path, missing_filename))
    
    return missing_files, sorted(list(all_years)), sorted(missing_file_paths)

def print_report(missing_files, all_years, missing_file_paths):
    """
    Prints a formatted report of missing files.
    """
    print(f"\nAnalysis Report")
    print(f"Years found in directory: {', '.join(all_years)}")
    print(f"\nLocations with missing files:")
    print("-" * 50)
    
    if not missing_files:
        print("No missing files found. All locations present in all years.")
        return
        
    for location, data in missing_files.items():
        print(f"\nLocation pattern: {location}")
        print(f"Missing in years: {', '.join(data['missing_years'])}")
        print("Existing files:")
        for year, filename in data['existing_files'].items():
            print(f"  {year}: {filename}")
    
    print("\nFull paths of missing files:")
    print("-" * 50)
    for path in missing_file_paths:
        print(path)

def main():
    # Replace this with your directory path
    directory_path = "/Users/diegobengochea/git/iberian.carbon/data/S2_PNOA_DATASET"
    
    try:
        missing_files, all_years, missing_file_paths = analyze_missing_files(directory_path)
        print_report(missing_files, all_years, missing_file_paths)
        
        # Optionally save missing file paths to a text file
        output_file = "missing_file_paths.txt"
        with open(output_file, 'w') as f:
            for path in missing_file_paths:
                f.write(f"{path}\n")
        print(f"\nMissing file paths have been saved to: {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()