# NYC Subway Crime Network Analysis

This repository contains the code for a computational social science study investigating the relationship between urban transit network structure, neighborhood socioeconomic conditions, and crime patterns in New York City.

## Project Overview

The study examines how network structure (centrality metrics), neighborhood disadvantage (poverty rates), and spatial heterogeneity interact to shape crime patterns across the NYC subway system. It uses network analysis, spatial statistics, and Generalized Additive Models (GAMs) to capture complex non-linear relationships.

## Repository Structure

```
subway-crime-network/
├── README.md                      # Project overview and instructions
├── requirements.txt               # Python dependencies
├── r_dependencies.R               # R dependencies
├── data/                          # Directory for datasets (not included in repo)
│   └── .gitkeep
├── scripts/
│   ├── python/
│   │   ├── 01_create_subway_network.py    # Creates subway network and calculates centrality metrics
│   │   └── 02_merge_crime_poverty.py      # Merges crime and poverty data with network metrics
│   └── r/
│       └── 03_statistical_analysis.R      # Performs GAM analysis
└── main.py                        # Main execution script
```

## Data Requirements

The analysis requires the following data files (not included in repository):
- `MTA_Subway_Stations_[date].csv`: NYC subway station data from MTA
- `NYC_Nhood ACS2008_12.shp`: Neighborhood Tabulation Areas with poverty data
- NYPD crime data files (pattern: `nypd_crimini_*.csv` or similar)

## Running the Analysis

1. Install Python dependencies:
```
pip install -r requirements.txt
```

2. Install R dependencies:
```
Rscript r_dependencies.R
```

3. Place required data files in the `data/` directory

4. Run the full pipeline:
```
python main.py
```

## Methodological Approach

The analysis follows these steps:
1. Subway network construction and centrality calculation
2. Spatial joining of crime incidents to nearest subway stations
3. Integration with neighborhood poverty data
4. Statistical modeling using GAMs with tensor product interactions
5. Analysis of network, socioeconomic, and spatial effects on crime patterns
