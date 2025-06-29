"""
NYC Subway Crime and Poverty Data Integration Module

This script integrates subway network data with crime incidents and 
neighborhood poverty data to create a comprehensive dataset for analysis.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.neighbors import BallTree
import glob
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_subway_complex_data(file_path):
    """Load subway complex data with centrality metrics"""
    logger.info(f"Loading subway complex data from {file_path}")
    complex_df = pd.read_csv(file_path)
    
    # Verify centrality metrics are present
    centrality_measures = ['Degree_Centrality', 'Betweenness_Centrality', 'Closeness_Centrality']
    missing_measures = [m for m in centrality_measures if m not in complex_df.columns]
    
    if missing_measures:
        logger.warning(f"Missing centrality measures: {missing_measures}")
    else:
        logger.info("All centrality measures present in complex data")
    
    # Remove Staten Island (if present)
    if 'Borough' in complex_df.columns:
        si_count = sum(complex_df['Borough'] == 'SI')
        if si_count > 0:
            complex_df = complex_df[complex_df['Borough'] != 'SI']
            logger.info(f"Removed {si_count} Staten Island stations")
    
    return complex_df

def load_crime_data(crime_patterns=None, data_dir='data'):
    """
    Load crime data from files matching patterns or create simulated data if no files found
    
    Args:
        crime_patterns: List of file patterns to search for (e.g., ['nypd_crimini_*.csv'])
        data_dir: Directory to search for crime data files
    
    Returns:
        DataFrame containing crime data
    """
    if crime_patterns is None:
        crime_patterns = ['nypd_crimini_*.csv', 'NYPD_*.csv', '*crime*.csv']
    
    # Search for files matching patterns
    crime_files = []
    for pattern in crime_patterns:
        pattern_path = os.path.join(data_dir, pattern)
        files = glob.glob(pattern_path)
        crime_files.extend(files)
    
    if crime_files:
        # Load real crime data
        df_list = []
        for file in crime_files:
            try:
                df = pd.read_csv(file, low_memory=False)
                df['source_file'] = os.path.basename(file)
                df_list.append(df)
                logger.info(f"Loaded {len(df)} records from {file}")
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        if df_list:
            df_crimes = pd.concat(df_list, ignore_index=True)
            logger.info(f"Combined {len(df_crimes)} crime records from {len(df_list)} files")
            return df_crimes
    
    # If no files found or loading failed, create simulated data
    logger.warning("No crime data files found. Creating simulated crime data.")
    return create_simulated_crime_data(load_subway_complex_data(os.path.join(data_dir, 'nyc_subway_complexes.csv')))

def create_simulated_crime_data(complex_df):
    """Create simulated crime data based on subway complex locations"""
    logger.info("Creating simulated crime data for demonstration purposes")
    np.random.seed(42)  # For reproducibility
    
    # Generate random crimes around subway stations
    fake_crimes = []
    
    for _, station in complex_df.iterrows():
        # Generate random number of crimes for this station
        num_crimes = np.random.randint(5, 50)
        
        for _ in range(num_crimes):
            # Generate random offset within 500m
            max_lat_offset = 0.005  # Approx 500m in latitude
            max_lon_offset = 0.005  # Approximation for longitude
            
            lat_offset = np.random.uniform(-max_lat_offset, max_lat_offset)
            lon_offset = np.random.uniform(-max_lon_offset, max_lon_offset)
            
            crime_lat = station['Latitude'] + lat_offset
            crime_lon = station['Longitude'] + lon_offset
            
            # Generate random crime type
            crime_type = np.random.choice(['FELONY', 'MISDEMEANOR', 'VIOLATION'], p=[0.3, 0.5, 0.2])
            
            # Generate random date and time
            year = np.random.randint(2020, 2025)
            month = np.random.randint(1, 13)
            day = np.random.randint(1, 29)
            hour = np.random.randint(0, 24)
            minute = np.random.randint(0, 60)
            
            fake_crimes.append({
                'latitude': crime_lat,
                'longitude': crime_lon,
                'law_cat_cd': crime_type,
                'cmplnt_fr_dt': f"{year}-{month:02d}-{day:02d}",
                'cmplnt_to_tm': f"{hour:02d}:{minute:02d}:00"
            })
    
    df_crimes = pd.DataFrame(fake_crimes)
    logger.info(f"Created {len(df_crimes)} simulated crime records")
    return df_crimes

def standardize_crime_coordinates(df_crimes):
    """Standardize coordinate column names and filter for valid coordinates"""
    # Identify coordinate columns
    lat_cols = ['Latitude', 'latitude', 'Lat', 'lat', 'LAT', 'LATITUDE', 'Y', 'y_coord']
    lon_cols = ['Longitude', 'longitude', 'Lon', 'lon', 'LONG', 'LON', 'LONGITUDE', 'X', 'x_coord']
    
    lat_col = None
    lon_col = None
    
    # Find latitude column
    for col in lat_cols:
        if col in df_crimes.columns:
            lat_col = col
            break
    
    # Find longitude column
    for col in lon_cols:
        if col in df_crimes.columns:
            lon_col = col
            break
    
    if lat_col is None or lon_col is None:
        logger.error("Could not identify coordinate columns in crime data")
        # Create placeholder columns
        df_crimes['latitude'] = np.nan
        df_crimes['longitude'] = np.nan
    else:
        # Standardize coordinate columns
        df_crimes['latitude'] = df_crimes[lat_col]
        df_crimes['longitude'] = df_crimes[lon_col]
    
    # Filter for valid coordinates
    df_filtered = df_crimes[df_crimes['latitude'].notna() & df_crimes['longitude'].notna()]
    df_filtered = df_filtered[df_filtered['latitude'] != 0.0]
    df_filtered = df_filtered[df_filtered['longitude'] != 0.0]
    
    # Filter for NYC coordinates
    df_filtered = df_filtered[df_filtered['latitude'].between(40.4, 41.0)]
    df_filtered = df_filtered[df_filtered['longitude'].between(-74.3, -73.5)]
    
    logger.info(f"Filtered to {len(df_filtered)} crime records with valid coordinates")
    return df_filtered

def add_time_dimensions(df_crimes):
    """Add time-based dimensions to crime data for analysis"""
    # Try to identify time column
    time_cols = ['cmplnt_to_tm', 'CMPLNT_TO_TM', 'complaint_time', 'COMPLAINT_TIME', 'time']
    time_col = next((col for col in time_cols if col in df_crimes.columns), None)
    
    if time_col:
        logger.info(f"Using '{time_col}' for temporal analysis")
        
        try:
            # Extract hour from time column
            df_crimes['hour'] = pd.to_datetime(df_crimes[time_col], errors='coerce').dt.hour
        except:
            try:
                # Alternative extraction if standard format fails
                df_crimes['hour'] = df_crimes[time_col].str.extract(r'^(\d{1,2})').astype(float)
            except:
                logger.warning(f"Could not extract hour from {time_col}. Creating random hours.")
                df_crimes['hour'] = np.random.randint(0, 24, size=len(df_crimes))
        
        # Create time period categories
        df_crimes['time_period'] = pd.cut(
            df_crimes['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)'],
            include_lowest=True
        )
        
        # Try to identify date column for day of week
        date_cols = ['cmplnt_fr_dt', 'CMPLNT_FR_DT', 'complaint_date', 'COMPLAINT_DATE', 'date']
        date_col = next((col for col in date_cols if col in df_crimes.columns), None)
        
        if date_col:
            try:
                df_crimes['date'] = pd.to_datetime(df_crimes[date_col], errors='coerce')
                df_crimes['day_of_week'] = df_crimes['date'].dt.day_name()
                df_crimes['is_weekend'] = df_crimes['day_of_week'].isin(['Saturday', 'Sunday'])
                logger.info(f"Added day of week information from '{date_col}'")
            except:
                logger.warning(f"Could not convert {date_col} to date format")
        
        logger.info("Added temporal dimensions to crime data")
    else:
        logger.warning("No time column found. Temporal analysis will not be available.")
    
    return df_crimes

def create_crime_geodataframe(df_crimes):
    """Convert crime dataframe to GeoDataFrame with point geometry"""
    gdf_crimes = gpd.GeoDataFrame(
        df_crimes,
        geometry=gpd.points_from_xy(df_crimes['longitude'], df_crimes['latitude']),
        crs="EPSG:4326"
    )
    return gdf_crimes

def create_complex_geodataframe(complex_df):
    """Convert subway complex dataframe to GeoDataFrame with point geometry"""
    # Identify coordinate columns
    lat_col = 'Latitude' if 'Latitude' in complex_df.columns else 'latitude'
    lon_col = 'Longitude' if 'Longitude' in complex_df.columns else 'longitude'
    
    gdf_complexes = gpd.GeoDataFrame(
        complex_df,
        geometry=gpd.points_from_xy(complex_df[lon_col], complex_df[lat_col]),
        crs="EPSG:4326"
    )
    return gdf_complexes

def assign_nearest_complex(gdf_crimes, gdf_complexes):
    """Assign each crime to its nearest subway complex using BallTree"""
    # Convert coordinates to radians for haversine distance calculation
    crime_coords = np.radians(gdf_crimes[['latitude', 'longitude']].values)
    
    lat_col = 'Latitude' if 'Latitude' in gdf_complexes.columns else 'latitude'
    lon_col = 'Longitude' if 'Longitude' in gdf_complexes.columns else 'longitude'
    complex_coords = np.radians(gdf_complexes[[lat_col, lon_col]].values)
    
    # Use BallTree for efficient nearest neighbor search
    tree = BallTree(complex_coords, metric='haversine')
    dist, idx = tree.query(crime_coords, k=1)
    
    # Add nearest complex information to crimes
    gdf_crimes['nearest_complex_index'] = idx.flatten()
    gdf_crimes['distance_km'] = dist.flatten() * 6371  # Earth radius in km
    
    # Extract complex identifiers
    complex_id_col = 'Complex_ID' if 'Complex_ID' in gdf_complexes.columns else 'complex_id'
    complex_name_col = 'Complex_Name' if 'Complex_Name' in gdf_complexes.columns else 'complex_name'
    
    gdf_crimes['nearest_complex_id'] = gdf_complexes.iloc[idx.flatten()][complex_id_col].values
    gdf_crimes['nearest_complex_name'] = gdf_complexes.iloc[idx.flatten()][complex_name_col].values
    
    logger.info("Assigned nearest subway complex to all crimes")
    return gdf_crimes

def filter_crimes_by_distance(gdf_crimes, max_distance_km=0.5):
    """Filter crimes to include only those within specified distance of a subway complex"""
    gdf_near_crimes = gdf_crimes[gdf_crimes['distance_km'] <= max_distance_km].copy()
    logger.info(f"Filtered to {len(gdf_near_crimes)} crimes within {max_distance_km}km of a subway complex")
    return gdf_near_crimes

def aggregate_crimes_by_complex(gdf_near_crimes):
    """Aggregate crimes by subway complex and legal category"""
    # Identify crime category column
    law_cat_cols = ['law_cat_cd', 'LAW_CAT_CD', 'crime_category', 'category', 'crime_type', 'ofns_desc', 'PD_DESC']
    law_cat_col = next((col for col in law_cat_cols if col in gdf_near_crimes.columns), None)
    
    if law_cat_col is None:
        logger.warning("No crime category column found. Creating placeholder category.")
        gdf_near_crimes['law_cat_cd'] = 'UNKNOWN'
        law_cat_col = 'law_cat_cd'
    
    # Group by complex and category
    complex_crime_counts = (
        gdf_near_crimes
        .groupby(['nearest_complex_id', law_cat_col])
        .size()
        .reset_index(name='crime_count')
    )
    
    # Create pivot table with crime types as columns
    complex_crime_pivot = complex_crime_counts.pivot_table(
        index='nearest_complex_id',
        columns=law_cat_col,
        values='crime_count',
        fill_value=0
    ).reset_index()
    
    # Reset column names
    complex_crime_pivot.columns.name = None
    
    # Ensure standard crime categories exist
    standard_categories = ['FELONY', 'MISDEMEANOR', 'VIOLATION']
    for category in standard_categories:
        if category not in complex_crime_pivot.columns:
            complex_crime_pivot[category] = 0
    
    # Standardize column names
    rename_dict = {
        'FELONY': 'felony_count',
        'MISDEMEANOR': 'misdemeanor_count',
        'VIOLATION': 'violation_count'
    }
    
    for old_name, new_name in rename_dict.items():
        if old_name in complex_crime_pivot.columns:
            complex_crime_pivot = complex_crime_pivot.rename(columns={old_name: new_name})
    
    # Add total crime count
    complex_crime_pivot['total_crime_count'] = 0
    for category in standard_categories:
        if category in rename_dict and rename_dict[category] in complex_crime_pivot.columns:
            complex_crime_pivot['total_crime_count'] += complex_crime_pivot[rename_dict[category]]
    
    logger.info(f"Created crime counts for {len(complex_crime_pivot)} subway complexes")
    return complex_crime_pivot

def aggregate_temporal_crime_data(gdf_near_crimes):
    """Create temporal aggregations of crimes if temporal dimensions exist"""
    temporal_aggregations = None
    
    if 'time_period' in gdf_near_crimes.columns:
        # Aggregate by time period
        time_period_counts = gdf_near_crimes.groupby(['nearest_complex_id', 'time_period']).size().unstack(fill_value=0)
        time_period_counts.columns = [f'crimes_{col.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")}' 
                                      for col in time_period_counts.columns]
        time_period_counts = time_period_counts.reset_index()
        
        # Aggregate by weekend vs weekday if available
        if 'is_weekend' in gdf_near_crimes.columns:
            weekend_counts = gdf_near_crimes.groupby(['nearest_complex_id', 'is_weekend']).size().unstack(fill_value=0)
            
            # Handle different column possibilities
            if True in weekend_counts.columns and False in weekend_counts.columns:
                weekend_counts.columns = ['crimes_weekday', 'crimes_weekend']
            elif True in weekend_counts.columns:
                weekend_counts.columns = ['crimes_weekend']
                weekend_counts['crimes_weekday'] = 0
            else:
                weekend_counts.columns = ['crimes_weekday']
                weekend_counts['crimes_weekend'] = 0
            
            weekend_counts = weekend_counts.reset_index()
            
            # Merge temporal aggregations
            temporal_aggregations = pd.merge(
                time_period_counts, 
                weekend_counts, 
                on='nearest_complex_id', 
                how='outer'
            )
        else:
            temporal_aggregations = time_period_counts
        
        # Fill NAs with 0
        temporal_aggregations = temporal_aggregations.fillna(0)
        logger.info("Created temporal crime aggregations")
    
    return temporal_aggregations

def load_poverty_data(nta_file, complex_df, data_dir='data'):
    """
    Load neighborhood poverty data from shapefile
    If file not found, create simulated poverty data
    """
    nta_path = os.path.join(data_dir, nta_file)
    
    try:
        if os.path.exists(nta_path):
            logger.info(f"Loading NTA shapefile from {nta_path}")
            nta = gpd.read_file(nta_path)
            
            # Calculate poverty percentage
            nta['poverty_pct'] = (nta['poor'] / nta['poptot']) * 100
            logger.info(f"Loaded {len(nta)} Neighborhood Tabulation Areas (NTAs)")
            
            # Create GeoDataFrame from complex data
            gdf_complexes = create_complex_geodataframe(complex_df)
            
            # Spatial join to assign NTA data to complexes
            logger.info("Performing spatial join to assign complexes to neighborhoods")
            complexes_with_nta = gpd.sjoin(
                gdf_complexes,
                nta[['ntacode', 'ntaname', 'poverty_pct', 'poptot', 'poor', 'geometry']],
                how='left',
                predicate='within'
            )
            
            # Extract poverty data
            poverty_data = complexes_with_nta[[
                'Complex_ID', 'ntacode', 'ntaname', 'poverty_pct', 'poptot', 'poor'
            ]]
            
            logger.info("Successfully added poverty data to subway complexes")
            return poverty_data
        else:
            logger.warning(f"NTA shapefile {nta_path} not found")
            return create_simulated_poverty_data(complex_df)
    except Exception as e:
        logger.error(f"Error loading poverty data: {e}")
        return create_simulated_poverty_data(complex_df)

def create_simulated_poverty_data(complex_df):
    """Create simulated poverty data for subway complexes"""
    logger.warning("Creating simulated poverty data")
    np.random.seed(42)  # For reproducibility
    
    # Create simulated NTA codes and names
    nta_codes = [f"NTA{i:03d}" for i in range(1, 51)]
    nta_names = [f"Neighborhood {i}" for i in range(1, 51)]
    
    # Assign random NTAs to complexes
    complex_df['ntacode'] = np.random.choice(nta_codes, size=len(complex_df))
    complex_df['ntaname'] = [nta_names[nta_codes.index(code)] for code in complex_df['ntacode']]
    
    # Generate simulated poverty data
    poverty_data = pd.DataFrame({
        'Complex_ID': complex_df['Complex_ID'],
        'ntacode': complex_df['ntacode'],
        'ntaname': complex_df['ntaname'],
        'poverty_pct': np.random.uniform(5, 35, size=len(complex_df)),  # 5% to 35%
        'poptot': np.random.randint(5000, 50000, size=len(complex_df)),
        'poor': [0] * len(complex_df)  # Will calculate after
    })
    
    # Calculate number of people in poverty
    poverty_data['poor'] = (poverty_data['poverty_pct'] / 100 * poverty_data['poptot']).astype(int)
    
    logger.info(f"Created simulated poverty data for {len(poverty_data)} complexes")
    return poverty_data

def create_integrated_dataset(complex_df, complex_crime_pivot, poverty_data, temporal_aggregations=None):
    """Integrate all data sources into a final analysis dataset"""
    # Merge crime data with complex data
    logger.info("Integrating all datasets")
    
    final_data = pd.merge(
        complex_df,
        complex_crime_pivot,
        left_on='Complex_ID',
        right_on='nearest_complex_id',
        how='left'
    )
    
    # Handle duplicate columns
    if 'nearest_complex_id' in final_data.columns:
        final_data = final_data.drop(columns=['nearest_complex_id'])
    
    # Fill missing crime counts with 0
    crime_cols = [col for col in final_data.columns if 'count' in col]
    for col in crime_cols:
        final_data[col] = final_data[col].fillna(0)
    
    # Merge with poverty data
    final_data = pd.merge(
        final_data,
        poverty_data,
        on='Complex_ID',
        how='left'
    )
    
    # Merge with temporal aggregations if available
    if temporal_aggregations is not None:
        final_data = pd.merge(
            final_data,
            temporal_aggregations,
            left_on='Complex_ID',
            right_on='nearest_complex_id',
            how='left'
        )
        
        # Remove duplicate column
        if 'nearest_complex_id' in final_data.columns:
            final_data = final_data.drop(columns=['nearest_complex_id'])
        
        # Fill missing temporal values with 0
        temporal_cols = [col for col in final_data.columns if col.startswith('crimes_')]
        for col in temporal_cols:
            final_data[col] = final_data[col].fillna(0)
    
    # Add derived variables for analysis
    # High centrality flag (top 25%)
    if 'Betweenness_Centrality' in final_data.columns:
        betweenness_col = 'Betweenness_Centrality'
        centrality_threshold = final_data[betweenness_col].quantile(0.75)
        final_data['high_centrality'] = final_data[betweenness_col] >= centrality_threshold
        
        # High crime flag (top 25%)
        if 'total_crime_count' in final_data.columns:
            crime_threshold = final_data['total_crime_count'].quantile(0.75)
            final_data['high_crime'] = final_data['total_crime_count'] >= crime_threshold
            
            # Create category for analysis
            final_data['centrality_crime_category'] = 'Other'
            final_data.loc[final_data['high_centrality'] & final_data['high_crime'], 'centrality_crime_category'] = 'High Centrality, High Crime'
            final_data.loc[final_data['high_centrality'] & ~final_data['high_crime'], 'centrality_crime_category'] = 'High Centrality, Low Crime'
            final_data.loc[~final_data['high_centrality'] & final_data['high_crime'], 'centrality_crime_category'] = 'Low Centrality, High Crime'
            final_data.loc[~final_data['high_centrality'] & ~final_data['high_crime'], 'centrality_crime_category'] = 'Low Centrality, Low Crime'
    
    # Calculate crime type ratios
    if all(col in final_data.columns for col in ['felony_count', 'misdemeanor_count', 'violation_count']):
        # Avoid division by zero
        total_crimes = final_data['felony_count'] + final_data['misdemeanor_count'] + final_data['violation_count']
        total_crimes = np.where(total_crimes == 0, 1, total_crimes)  # Replace zeros with one
        
        final_data['felony_ratio'] = final_data['felony_count'] / total_crimes
        final_data['misdemeanor_ratio'] = final_data['misdemeanor_count'] / total_crimes
        final_data['violation_ratio'] = final_data['violation_count'] / total_crimes
    
    # Add standardized centrality measures for analysis
    if 'Degree_Centrality' in final_data.columns:
        final_data['degree_std'] = (final_data['Degree_Centrality'] - final_data['Degree_Centrality'].mean()) / final_data['Degree_Centrality'].std()
    
    if 'Betweenness_Centrality' in final_data.columns:
        final_data['betweenness_std'] = (final_data['Betweenness_Centrality'] - final_data['Betweenness_Centrality'].mean()) / final_data['Betweenness_Centrality'].std()
    
    if 'Closeness_Centrality' in final_data.columns:
        final_data['closeness_std'] = (final_data['Closeness_Centrality'] - final_data['Closeness_Centrality'].mean()) / final_data['Closeness_Centrality'].std()
    
    # Add log-transformed population for modeling
    final_data['log_poptot'] = np.log(final_data['poptot'].replace(0, 1))
    
    # Add poverty binary indicator
    if 'poverty_pct' in final_data.columns:
        poverty_median = final_data['poverty_pct'].median()
        final_data['poverty_high'] = np.where(final_data['poverty_pct'] > poverty_median, 1, 0)
    
    logger.info(f"Created integrated dataset with {len(final_data)} records and {len(final_data.columns)} variables")
    return final_data

def integrate_subway_crime_poverty_data(
    complex_file='data/nyc_subway_complexes.csv',
    nta_file='nycnhood_acs/NYC_Nhood ACS2008_12.shp',
    output_file='data/nyc_subway_crime_poverty_analysis.csv',
    crime_patterns=None,
    data_dir='data'
):
    """Main function to integrate all data sources"""
    # Load subway complex data
    complex_df = load_subway_complex_data(complex_file)
    
    # Load crime data
    df_crimes = load_crime_data(crime_patterns, data_dir)
    
    # Standardize crime coordinates
    df_crimes = standardize_crime_coordinates(df_crimes)
    
    # Add time dimensions
    df_crimes = add_time_dimensions(df_crimes)
    
    # Convert to GeoDataFrames
    gdf_crimes = create_crime_geodataframe(df_crimes)
    gdf_complexes = create_complex_geodataframe(complex_df)
    
    # Assign nearest complex to each crime
    gdf_crimes = assign_nearest_complex(gdf_crimes, gdf_complexes)
    
    # Filter to crimes within 500m of subway complex
    gdf_near_crimes = filter_crimes_by_distance(gdf_crimes)
    
    # Aggregate crimes by complex and category
    complex_crime_pivot = aggregate_crimes_by_complex(gdf_near_crimes)
    
    # Create temporal aggregations if available
    temporal_aggregations = aggregate_temporal_crime_data(gdf_near_crimes)
    
    # Load poverty data
    poverty_data = load_poverty_data(nta_file, complex_df, data_dir)
    
    # Create integrated dataset
    final_data = create_integrated_dataset(complex_df, complex_crime_pivot, poverty_data, temporal_aggregations)
    
    # Save final dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_data.to_csv(output_file, index=False)
    logger.info(f"Saved integrated dataset to {output_file}")
    
    return final_data

if __name__ == "__main__":
    # This allows the script to be run directly for testing
    integrate_subway_crime_poverty_data()
