"""
NYC Subway Network Construction Module

This script creates a network representation of the NYC subway system,
calculates centrality metrics, and saves the results for further analysis.
"""

import networkx as nx
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from collections import Counter
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def load_station_data(file_path):
    """Load subway station data and identify column names"""
    logger.info(f"Loading subway station data from {file_path}")
    stations_df = pd.read_csv(file_path)
    
    # Identify column names based on conventions in the file
    column_mapping = {
        'Station ID': ['Station ID', 'STATION_ID'],
        'Complex ID': ['Complex ID', 'COMPLEX_ID'],
        'Stop Name': ['Stop Name', 'STOP_NAME'],
        'Borough': ['Borough', 'BOROUGH'],
        'Daytime Routes': ['Daytime Routes', 'DAYTIME_ROUTES'],
        'Line': ['Line', 'LINE'],
        'GTFS Latitude': ['GTFS Latitude', 'GTFS_LATITUDE'],
        'GTFS Longitude': ['GTFS Longitude', 'GTFS_LONGITUDE'],
        'GTFS Stop ID': ['GTFS Stop ID', 'GTFS_STOP_ID']
    }
    
    # Determine actual column names in the dataset
    actual_columns = {}
    for std_name, possible_names in column_mapping.items():
        for name in possible_names:
            if name in stations_df.columns:
                actual_columns[std_name] = name
                break
        if std_name not in actual_columns:
            logger.warning(f"Column {std_name} not found in dataset")
    
    return stations_df, actual_columns

def create_complex_dataframe(stations_df, columns):
    """Create a dataframe of station complexes with aggregated information"""
    logger.info("Creating unified complex dataframe")
    
    # Group stations by Complex ID
    complex_groups = stations_df.groupby(columns['Complex ID'])
    complex_data = []
    
    for complex_id, group in complex_groups:
        # Calculate average coordinates
        avg_lat = group[columns['GTFS Latitude']].mean()
        avg_lon = group[columns['GTFS Longitude']].mean()
        
        # Collect all routes serving this complex
        all_routes = []
        for _, station in group.iterrows():
            routes = str(station[columns['Daytime Routes']]).split()
            all_routes.extend(routes)
        unique_routes = sorted(set(all_routes))
        
        # Determine main borough of the complex
        borough_counts = Counter(group[columns['Borough']])
        main_borough = borough_counts.most_common(1)[0][0]
        
        # Collect all lines passing through this complex
        all_lines = group[columns['Line']].unique().tolist()
        
        # Collect all station IDs in this complex
        station_ids = group[columns['Station ID']].tolist()
        gtfs_stop_ids = []
        if 'GTFS Stop ID' in columns:
            gtfs_stop_ids = group[columns['GTFS Stop ID']].tolist()
        
        # Collect station names
        station_names = group[columns['Stop Name']].tolist()
        
        # Create complex name
        if len(group) > 1:
            complex_name = f"{group[columns['Stop Name']].values[0]} (Complex)"
        else:
            complex_name = group[columns['Stop Name']].values[0]
        
        # Add complex data
        complex_data.append({
            'Complex_ID': complex_id,
            'Complex_Name': complex_name,
            'Borough': main_borough,
            'Latitude': avg_lat,
            'Longitude': avg_lon,
            'Routes': ", ".join(unique_routes),
            'Lines': ", ".join(all_lines),
            'Station_Count': len(group),
            'Station_IDs': ", ".join(map(str, station_ids)),
            'GTFS_Stop_IDs': ", ".join(map(str, gtfs_stop_ids)),
            'Station_Names': " | ".join(station_names)
        })
    
    complex_df = pd.DataFrame(complex_data)
    logger.info(f"Created {len(complex_df)} unified complexes (instead of {len(stations_df)} individual stations)")
    return complex_df

def create_subway_network(complex_df, stations_df, columns):
    """Create a network graph of subway complexes and calculate routes between them"""
    logger.info("Creating subway network graph")
    
    # Initialize graph
    G_complex = nx.Graph()
    
    # Add nodes for each complex
    for _, complex_info in complex_df.iterrows():
        G_complex.add_node(
            complex_info['Complex_ID'],
            name=complex_info['Complex_Name'],
            borough=complex_info['Borough'],
            lat=complex_info['Latitude'],
            lon=complex_info['Longitude'],
            routes=complex_info['Routes'],
            lines=complex_info['Lines'],
            station_count=complex_info['Station_Count'],
            station_ids=complex_info['Station_IDs'].split(", "),
            gtfs_stop_ids=complex_info['GTFS_Stop_IDs'].split(", ")
        )
    
    logger.info(f"Added {len(G_complex.nodes())} nodes to the graph (station complexes)")
    
    # Create a mapping from Station ID to Complex ID
    station_to_complex = dict(zip(stations_df[columns['Station ID']], 
                                  stations_df[columns['Complex ID']]))
    
    # Group stations by route
    route_stations = {}
    for _, station in stations_df.iterrows():
        routes = str(station[columns['Daytime Routes']]).split()
        for route in routes:
            if route not in route_stations:
                route_stations[route] = []
            route_stations[route].append(station[columns['Station ID']])
    
    # Add edges based on routes
    logger.info("Adding connections based on routes")
    
    for route, stations in route_stations.items():
        # Convert stations to complexes
        complexes = [station_to_complex[station] for station in stations]
        # Remove duplicates (stations in same complex)
        unique_complexes = []
        for complex_id in complexes:
            if complex_id not in unique_complexes:
                unique_complexes.append(complex_id)
        
        if len(unique_complexes) <= 1:
            continue
        
        # Order complexes for this route using nearest neighbor approach
        ordered_complexes = []
        remaining_complexes = set(unique_complexes)
        
        # Choose northernmost complex as starting point
        start_complex = max(remaining_complexes, 
                           key=lambda c: G_complex.nodes[c]['lat'])
        ordered_complexes.append(start_complex)
        remaining_complexes.remove(start_complex)
        
        current_complex = start_complex
        while remaining_complexes:
            # Find closest complex to current one
            min_dist = float('inf')
            nearest = None
            
            current_lat = G_complex.nodes[current_complex]['lat']
            current_lon = G_complex.nodes[current_complex]['lon']
            
            for complex_id in remaining_complexes:
                complex_lat = G_complex.nodes[complex_id]['lat']
                complex_lon = G_complex.nodes[complex_id]['lon']
                
                dist = haversine(current_lon, current_lat, complex_lon, complex_lat)
                if dist < min_dist:
                    min_dist = dist
                    nearest = complex_id
            
            # Add nearest complex to order
            if nearest:
                ordered_complexes.append(nearest)
                remaining_complexes.remove(nearest)
                current_complex = nearest
            else:
                break
        
        # Connect consecutive complexes
        for i in range(len(ordered_complexes) - 1):
            complex1 = ordered_complexes[i]
            complex2 = ordered_complexes[i + 1]
            
            # Calculate distance
            complex1_lat = G_complex.nodes[complex1]['lat']
            complex1_lon = G_complex.nodes[complex1]['lon']
            complex2_lat = G_complex.nodes[complex2]['lat']
            complex2_lon = G_complex.nodes[complex2]['lon']
            
            dist = haversine(complex1_lon, complex1_lat, complex2_lon, complex2_lat)
            
            # Add edge if distance is reasonable (< 8km)
            if dist < 8.0:
                # If edge already exists, update routes
                if G_complex.has_edge(complex1, complex2):
                    existing_routes = G_complex[complex1][complex2].get('routes', '').split(', ')
                    if route not in existing_routes:
                        existing_routes.append(route)
                        G_complex[complex1][complex2]['routes'] = ', '.join(existing_routes)
                else:
                    # Create new edge
                    G_complex.add_edge(complex1, complex2, type='route', routes=route, distance=dist)
    
    logger.info(f"Added {G_complex.number_of_edges()} edges based on train routes")
    
    # Add special connections that might be missed
    add_special_connections(G_complex)
    
    # Remove Staten Island from analysis
    si_nodes = [node for node in G_complex.nodes() if G_complex.nodes[node]['borough'] == 'SI']
    if si_nodes:
        logger.info(f"Removing {len(si_nodes)} Staten Island nodes from analysis")
        G_complex.remove_nodes_from(si_nodes)
    
    # Ensure network is connected
    ensure_network_connectivity(G_complex)
    
    return G_complex

def add_special_connections(G):
    """Add special connections that might be missed in the route-based approach"""
    # Add connection between Broad Channel and Howard Beach-JFK Airport
    broad_channel = None
    howard_beach = None
    
    for node in G.nodes():
        if "Broad Channel" in G.nodes[node]['name']:
            broad_channel = node
        elif "Howard Beach" in G.nodes[node]['name']:
            howard_beach = node
    
    if broad_channel and howard_beach:
        if not G.has_edge(broad_channel, howard_beach):
            dist = haversine(
                G.nodes[broad_channel]['lon'], G.nodes[broad_channel]['lat'],
                G.nodes[howard_beach]['lon'], G.nodes[howard_beach]['lat']
            )
            G.add_edge(broad_channel, howard_beach, type='special_route', routes='A', distance=dist)
            logger.info(f"Added special connection between Broad Channel and Howard Beach (distance: {dist:.3f} km)")

def ensure_network_connectivity(G):
    """Ensure the network is connected by adding edges between disconnected components"""
    components = list(nx.connected_components(G))
    logger.info(f"Network has {len(components)} connected components")
    
    if len(components) > 1:
        # Sort components by size (largest first)
        components.sort(key=len, reverse=True)
        main_component = list(components[0])
        
        # Connect smaller components to main component
        for i, component in enumerate(components[1:], 1):
            logger.info(f"Connecting component {i} ({len(component)} nodes) to main component")
            
            # Find closest pair of nodes between components
            min_dist = float('inf')
            nearest_pair = (None, None)
            
            for node1 in component:
                for node2 in main_component:
                    dist = haversine(
                        G.nodes[node1]['lon'], G.nodes[node1]['lat'],
                        G.nodes[node2]['lon'], G.nodes[node2]['lat']
                    )
                    if dist < min_dist:
                        min_dist = dist
                        nearest_pair = (node1, node2)
            
            # Add connection
            if nearest_pair[0] and nearest_pair[1]:
                G.add_edge(nearest_pair[0], nearest_pair[1], type='manual_fix', distance=min_dist)
                logger.info(f"Added connection between {G.nodes[nearest_pair[0]]['name']} and "
                          f"{G.nodes[nearest_pair[1]]['name']} (distance: {min_dist:.3f} km)")

def calculate_centrality_metrics(G):
    """Calculate network centrality metrics for all nodes"""
    logger.info("Calculating network centrality metrics")
    
    # Calculate centrality metrics
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    
    # Add metrics to graph nodes
    for node in G.nodes():
        G.nodes[node]['degree_centrality'] = degree_centrality.get(node, 0)
        G.nodes[node]['betweenness_centrality'] = betweenness_centrality.get(node, 0)
        G.nodes[node]['closeness_centrality'] = closeness_centrality.get(node, 0)
    
    return G

def create_edge_dataframe(G):
    """Create a dataframe of network edges"""
    edge_data = []
    
    for source, target, data in G.edges(data=True):
        edge = {
            'Source_Complex_ID': source,
            'Source_Complex_Name': G.nodes[source]['name'],
            'Target_Complex_ID': target,
            'Target_Complex_Name': G.nodes[target]['name'],
            'Edge_Type': data.get('type', 'unknown'),
            'Routes': data.get('routes', ''),
            'Distance_km': data.get('distance', 0)
        }
        edge_data.append(edge)
    
    return pd.DataFrame(edge_data)

def add_centrality_to_complex_df(complex_df, G):
    """Add centrality metrics to the complex dataframe"""
    for index, row in complex_df.iterrows():
        complex_id = row['Complex_ID']
        if complex_id in G.nodes():
            complex_df.at[index, 'Degree_Centrality'] = G.nodes[complex_id]['degree_centrality']
            complex_df.at[index, 'Betweenness_Centrality'] = G.nodes[complex_id]['betweenness_centrality']
            complex_df.at[index, 'Closeness_Centrality'] = G.nodes[complex_id]['closeness_centrality']
    
    return complex_df

def create_subway_network_data(subway_stations_file, output_dir='data'):
    """Main function to create subway network and save results"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load station data
    stations_df, columns = load_station_data(subway_stations_file)
    
    # Create complex dataframe
    complex_df = create_complex_dataframe(stations_df, columns)
    
    # Create network
    G = create_subway_network(complex_df, stations_df, columns)
    
    # Calculate centrality metrics
    G = calculate_centrality_metrics(G)
    
    # Add centrality metrics to complex dataframe
    complex_df = add_centrality_to_complex_df(complex_df, G)
    
    # Create edge dataframe
    edges_df = create_edge_dataframe(G)
    
    # Save results
    complex_output_path = os.path.join(output_dir, 'nyc_subway_complexes.csv')
    edges_output_path = os.path.join(output_dir, 'nyc_subway_edges.csv')
    
    complex_df.to_csv(complex_output_path, index=False)
    edges_df.to_csv(edges_output_path, index=False)
    
    logger.info(f"Saved complex data to {complex_output_path}")
    logger.info(f"Saved edge data to {edges_output_path}")
    
    return G, complex_df, edges_df

if __name__ == "__main__":
    # This allows the script to be run directly for testing
    subway_stations_file = "data/MTA_Subway_Stations_20250530.csv"
    create_subway_network_data(subway_stations_file)
