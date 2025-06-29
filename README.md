# Urban Crime and Transit Networks: A Computational Social Science Approach

This repository contains the code for a computational social science study investigating how network structure, neighborhood disadvantage, and spatial heterogeneity interact to shape crime patterns across urban transit systems, using New York City's subway network as a case study.

## Research Context

Urban crime exhibits complex spatial patterns shaped by the interplay of infrastructure, socioeconomic conditions, and place-based social processes. This study integrates network analysis with advanced statistical modeling to examine three criminological frameworks:

- **Routine Activity Theory**: How transit nodes create convergence spaces for potential offenders and targets
- **Broken Windows Theory**: How neighborhood disadvantage and disorder influence criminal opportunities
- **Collective Efficacy Theory**: How similar socioeconomic conditions can yield different outcomes in different spatial contexts

## Key Findings

Our analysis reveals three major patterns:

1. **Network Effects**: Degree centrality consistently predicts higher crime rates across all crime types (18.5-25.9% increase per standard deviation), demonstrating that a station's network position influences crime independently of neighborhood characteristics.

2. **Non-linear Socioeconomic Effects**: The relationship between poverty and crime is highly non-linear (EDF: 11.45-12.13) and spatially heterogeneous (EDF: 24.51-26.84), suggesting threshold effects and place-specific social dynamics that conventional linear models fail to capture.

3. **Borough-Level Variations**: Substantial differences persist even after controlling for network and socioeconomic factors, with the Bronx showing 52-63% higher crime rates than Brooklyn.

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
│       └── 03_statistical_analysis.R      # Performs GAM analysis with tensor product interactions
└── main.py                        # Main execution script
```

## Data Requirements

The analysis requires the following data files (not included in repository):
- `MTA_Subway_Stations_[date].csv`: NYC subway station data (2008-2012)
- `NYC_Nhood ACS2008_12.shp`: Neighborhood Tabulation Areas with poverty data
- NYPD crime data files with geocoded incident locations

## Methodological Approach

Our computational approach involves three integrated components:

1. **Network Construction**: We model the NYC subway system (423 complexes, 518 edges) as an undirected graph and calculate degree, betweenness, and closeness centrality measures using NetworkX.

2. **Spatial Data Integration**: We spatially join crime incidents to their nearest subway complex (500m radius), and integrate neighborhood poverty data through spatial overlay operations.

3. **Statistical Modeling**: We employ Generalized Additive Models (GAMs) with Negative Binomial distribution to account for overdispersion in crime counts. Our models include:
   - Parametric terms for network centrality and borough effects
   - Smooth functions for non-linear poverty effects
   - Tensor product interactions to capture spatially varying relationships

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

## Citation

If using this code or methodology in your research, please cite:

Attolico, L.D., Mauro, A., Rossi, M., Sartore, T., & Stanghellini, F. (2025). Urban Crime and Transit Networks: A Computational Social Science Approach. University of Trento.