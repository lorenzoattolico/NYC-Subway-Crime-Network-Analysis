"""
NYC Subway Crime Network Analysis - Main Script

This script orchestrates the full analysis pipeline for examining the relationship
between subway network structure, socioeconomic conditions, and crime patterns
in New York City.

The pipeline consists of three main stages:
1. Subway network construction and centrality calculation
2. Integration of crime data and poverty metrics
3. Statistical analysis using Generalized Additive Models (GAMs)
"""

import os
import sys
import logging
import argparse
import subprocess
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("subway_crime_analysis.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def create_directory_structure():
    """Create the necessary directory structure for the project"""
    directories = ['data', 'results', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def run_subway_network_analysis(stations_file, output_dir='data'):
    """Run the subway network construction and analysis"""
    logger.info("Starting subway network construction and analysis")
    
    try:
        # Import the module
        sys.path.append('scripts/python')
        from scripts.python.01_create_subway_network import create_subway_network_data
        
        # Run the analysis
        G, complex_df, edges_df = create_subway_network_data(stations_file, output_dir)
        logger.info(f"Subway network analysis completed. Created {len(complex_df)} complexes and {len(edges_df)} edges.")
        return True
    except Exception as e:
        logger.error(f"Error in subway network analysis: {e}")
        return False

def run_crime_poverty_integration(complex_file, nta_file, output_file, crime_patterns=None, data_dir='data'):
    """Run the integration of crime and poverty data with subway network"""
    logger.info("Starting integration of crime and poverty data")
    
    try:
        # Import the module
        sys.path.append('scripts/python')
        from scripts.python.02_merge_crime_poverty import integrate_subway_crime_poverty_data
        
        # Run the integration
        final_data = integrate_subway_crime_poverty_data(
            complex_file=complex_file,
            nta_file=nta_file,
            output_file=output_file,
            crime_patterns=crime_patterns,
            data_dir=data_dir
        )
        logger.info(f"Data integration completed. Created dataset with {len(final_data)} records.")
        return True
    except Exception as e:
        logger.error(f"Error in crime and poverty data integration: {e}")
        return False

def run_statistical_analysis(input_file, output_dir='results'):
    """Run the statistical analysis using R and GAMs"""
    logger.info("Starting statistical analysis with GAMs in R")
    
    r_script_path = os.path.abspath('scripts/r/03_statistical_analysis.R')
    
    if not os.path.exists(r_script_path):
        logger.error(f"R script not found at {r_script_path}")
        return False
    
    try:
        # Build the R command
        r_command = [
            "Rscript",
            r_script_path,
            input_file,
            output_dir
        ]
        
        # Run the R script as a subprocess
        logger.info(f"Running R script: {' '.join(r_command)}")
        
        # Instead of directly using R from Python, call it as a subprocess
        process = subprocess.Popen(
            ["Rscript", "-e", f"source('{r_script_path}'); run_statistical_analysis('{input_file}', '{output_dir}')"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"R script failed with return code {process.returncode}")
            logger.error(f"STDERR: {stderr}")
            return False
        
        logger.info("Statistical analysis completed successfully")
        logger.debug(f"R script output: {stdout}")
        return True
    except Exception as e:
        logger.error(f"Error running statistical analysis: {e}")
        return False

def main():
    """Main function to run the complete analysis pipeline"""
    parser = argparse.ArgumentParser(description='NYC Subway Crime Network Analysis')
    
    parser.add_argument('--stations', type=str, default='data/MTA_Subway_Stations_20250530.csv',
                        help='Path to subway stations data file')
    parser.add_argument('--nta', type=str, default='data/nycnhood_acs/NYC_Nhood ACS2008_12.shp',
                        help='Path to neighborhood tabulation areas shapefile')
    parser.add_argument('--crime-pattern', type=str, action='append',
                        help='File pattern for crime data files (can be used multiple times)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory for input/output data files')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory for analysis results')
    parser.add_argument('--skip-network', action='store_true',
                        help='Skip subway network construction stage')
    parser.add_argument('--skip-integration', action='store_true',
                        help='Skip crime and poverty data integration stage')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='Skip statistical analysis stage')
    
    args = parser.parse_args()
    
    # Create directory structure
    create_directory_structure()
    
    start_time = datetime.now()
    logger.info(f"Starting analysis pipeline at {start_time}")
    
    # Define file paths
    complex_file = os.path.join(args.data_dir, 'nyc_subway_complexes.csv')
    edges_file = os.path.join(args.data_dir, 'nyc_subway_edges.csv')
    integrated_file = os.path.join(args.data_dir, 'nyc_subway_crime_poverty_analysis.csv')
    
    # Stage 1: Subway Network Construction
    if not args.skip_network:
        logger.info("STAGE 1: Subway Network Construction")
        if not os.path.exists(args.stations):
            logger.error(f"Subway stations file not found: {args.stations}")
            return
        
        success = run_subway_network_analysis(args.stations, args.data_dir)
        if not success:
            logger.error("Subway network construction failed. Cannot proceed.")
            return
    else:
        logger.info("STAGE 1: Subway Network Construction [SKIPPED]")
        if not os.path.exists(complex_file) or not os.path.exists(edges_file):
            logger.warning(f"Required network files not found: {complex_file} or {edges_file}")
    
    # Stage 2: Crime and Poverty Data Integration
    if not args.skip_integration:
        logger.info("STAGE 2: Crime and Poverty Data Integration")
        if not os.path.exists(complex_file):
            logger.error(f"Complex file not found: {complex_file}")
            return
        
        success = run_crime_poverty_integration(
            complex_file=complex_file,
            nta_file=args.nta,
            output_file=integrated_file,
            crime_patterns=args.crime_pattern,
            data_dir=args.data_dir
        )
        if not success:
            logger.error("Crime and poverty data integration failed. Cannot proceed.")
            return
    else:
        logger.info("STAGE 2: Crime and Poverty Data Integration [SKIPPED]")
        if not os.path.exists(integrated_file):
            logger.warning(f"Required integrated data file not found: {integrated_file}")
    
    # Stage 3: Statistical Analysis
    if not args.skip_analysis:
        logger.info("STAGE 3: Statistical Analysis")
        if not os.path.exists(integrated_file):
            logger.error(f"Integrated data file not found: {integrated_file}")
            return
        
        success = run_statistical_analysis(integrated_file, args.results_dir)
        if not success:
            logger.error("Statistical analysis failed.")
            return
    else:
        logger.info("STAGE 3: Statistical Analysis [SKIPPED]")
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Analysis pipeline completed at {end_time}")
    logger.info(f"Total duration: {duration}")

if __name__ == "__main__":
    main()
