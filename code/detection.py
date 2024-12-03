import numpy as np
import pandas as pd
from scipy.ndimage import label
from sklearn.cluster import DBSCAN
from pyproj import Proj, transform
from scipy.ndimage import gaussian_filter, median_filter
from input_output import load_data

def apply_smoothing(prec, method='gaussian', sigma=1, size=6):
    """
    Apply smoothing to the precipitation field.

    Parameters:
    - prec: 2D array of precipitation values.
    - method: Smoothing method ('gaussian' or 'median').
    - sigma: Standard deviation for Gaussian kernel (for 'gaussian' method).
    - size: Size of the neighborhood (for 'median' method).

    Returns:
    - prec_smooth: Smoothed precipitation field.
    """
    

    if method == 'gaussian':
        prec_smooth = gaussian_filter(prec, sigma=sigma)
    elif method == 'median':
        prec_smooth = median_filter(prec, size=size)
    else:
        raise ValueError("Invalid method. Choose 'gaussian' or 'median'.")
    return prec_smooth

def identify_moderate_precipitation(prec, moderate_prec_threshold):
    """
    Create a mask of grid points with moderate precipitation.
    """
    moderate_prec_mask = prec >= moderate_prec_threshold
    return moderate_prec_mask

def cluster_moderate_precipitation(moderate_prec_mask, lat, lon, eps_km, min_samples):
    """
    Use DBSCAN to cluster moderate precipitation grid points.
    """
    # Get indices of moderate precipitation points
    moderate_indices = np.where(moderate_prec_mask)
    moderate_lat = lat.values[moderate_indices]
    moderate_lon = lon.values[moderate_indices]

    # Convert lat/lon to projected coordinates (e.g., UTM)
    proj_wgs84 = Proj('epsg:4326')
    proj_utm = Proj('epsg:32718')  # Adjust to your UTM zone

    # Convert coordinates
    moderate_x, moderate_y = transform(proj_wgs84, proj_utm, moderate_lon, moderate_lat)

    # Stack coordinates for clustering
    coords = np.column_stack((moderate_x, moderate_y))

    # Convert eps from km to meters
    eps_meters = eps_km * 1000

    # Perform DBSCAN clustering
    db = DBSCAN(eps=eps_meters, min_samples=min_samples, metric='euclidean').fit(coords)
    labels = db.labels_

    # Create an empty array for labeled regions
    moderate_labeled_regions = np.zeros_like(moderate_prec_mask, dtype=int)

    # Assign labels to the moderate precipitation mask
    for idx, (i, j) in enumerate(zip(moderate_indices[0], moderate_indices[1])):
        if labels[idx] != -1:
            moderate_labeled_regions[i, j] = labels[idx] + 1  # labels start from 0

    return moderate_labeled_regions

def cluster_precipitation_regions(moderate_prec_mask):
    """
    Cluster moderate precipitation regions using connected-component labeling.
    """
    structure = np.array([[1,1,1],
                          [1,1,1],
                          [1,1,1]])  # 8-connectivity
    labeled_regions, num_features = label(moderate_prec_mask, structure=structure)
    return labeled_regions, num_features

def filter_clusters_by_size(labeled_regions, min_area_km2, grid_spacing_km):
    """
    Filter clusters based on the minimum area threshold.
    """
   
    cluster_labels = np.unique(labeled_regions)
    cluster_labels = cluster_labels[cluster_labels != 0]  # Exclude background label 0

    # Area of one grid cell
    grid_cell_area_km2 = grid_spacing_km ** 2

    # Dictionary to store cluster sizes
    cluster_sizes = {}

    for label in cluster_labels:
        cluster_mask = labeled_regions == label
        cluster_area_km2 = np.sum(cluster_mask) * grid_cell_area_km2
        if cluster_area_km2 >= min_area_km2:
            cluster_sizes[label] = cluster_area_km2

    # Create a mask for clusters to keep
    filtered_labeled_regions = np.where(np.isin(labeled_regions, list(cluster_sizes.keys())), labeled_regions, 0)
    return filtered_labeled_regions

def filter_clusters_with_heavy_precipitation(filtered_labeled_regions, precipitation, heavy_prec_threshold):
    """
    Keep clusters that contain at least one heavy precipitation grid point.
    """
  
    cluster_labels = np.unique(filtered_labeled_regions)
    cluster_labels = cluster_labels[cluster_labels != 0]  # Exclude background label 0

    clusters_to_keep = []

    for label in cluster_labels:
        cluster_mask = filtered_labeled_regions == label
        heavy_prec_in_cluster = np.any(np.logical_and(cluster_mask, precipitation >= heavy_prec_threshold))
        if heavy_prec_in_cluster:
            clusters_to_keep.append(label)

    # Create a final mask for clusters to keep
    final_labeled_regions = np.where(np.isin(filtered_labeled_regions, clusters_to_keep), filtered_labeled_regions, 0)
    return final_labeled_regions


def calculate_cluster_areas(labeled_regions, grid_spacing_km):
    """
    Calculate the area of each cluster based on the number of grid cells and grid spacing.
    """
    cluster_labels = np.unique(labeled_regions)
    cluster_labels = cluster_labels[cluster_labels != 0]  # Exclude background label 0

    # Area of one grid cell
    grid_cell_area_km2 = grid_spacing_km ** 2

    # Dictionary to store areas
    cluster_areas = {}

    for label_id in cluster_labels:
        cluster_mask = labeled_regions == label_id
        num_grid_cells = np.sum(cluster_mask)
        area_km2 = num_grid_cells * grid_cell_area_km2
        cluster_areas[label_id] = area_km2

    return cluster_areas

def filter_clusters_by_area(labeled_regions, cluster_areas, min_area_km2):
    """
    Filter clusters based on the minimum area threshold.
    """
    clusters_to_keep = [label_id for label_id, area in cluster_areas.items() if area >= min_area_km2]

    # Create a mask for clusters to keep
    filtered_labeled_regions = np.where(np.isin(labeled_regions, clusters_to_keep), labeled_regions, 0)
    return filtered_labeled_regions

def calculate_heavy_precip_percentage(filtered_labeled_regions, prec, heavy_prec_threshold):
    """
    Calculate the percentage of heavy precipitation grid cells within each cluster.
    """
    cluster_labels = np.unique(filtered_labeled_regions)
    cluster_labels = cluster_labels[cluster_labels != 0]

    # Dictionary to store heavy precipitation percentages
    heavy_prec_percentages = {}

    for label_id in cluster_labels:
        cluster_mask = filtered_labeled_regions == label_id
        total_cells = np.sum(cluster_mask)
        # Identify heavy precipitation cells within the cluster
        heavy_prec_mask = (prec >= heavy_prec_threshold) & cluster_mask
        heavy_cells = np.sum(heavy_prec_mask)
        # Calculate percentage
        heavy_prec_percentage = (heavy_cells / total_cells) * 100
        heavy_prec_percentages[label_id] = heavy_prec_percentage

    return heavy_prec_percentages

def select_clusters_by_heavy_precip_percentage(filtered_labeled_regions, heavy_prec_percentages, min_heavy_prec_percentage):
    """
    Select clusters where the heavy precipitation percentage exceeds the threshold.
    """
    clusters_to_keep = [label_id for label_id, percentage in heavy_prec_percentages.items() if percentage >= min_heavy_prec_percentage]

    # Create final mask
    final_labeled_regions = np.where(np.isin(filtered_labeled_regions, clusters_to_keep), filtered_labeled_regions, 0)
    return final_labeled_regions

def create_mcs_dataframe(final_labeled_regions, lat, lon, precipitation):
    """
    Create a DataFrame of the final MCS regions.
    """

    mcs_indices = np.where(final_labeled_regions > 0)
    mcs_lat = lat.values[mcs_indices]
    mcs_lon = lon.values[mcs_indices]
    mcs_prec_values = precipitation[mcs_indices]
    mcs_region_labels = final_labeled_regions[mcs_indices]

    df_mcs = pd.DataFrame({
        'lat': mcs_lat,
        'lon': mcs_lon,
        'precip': mcs_prec_values,
        'region_label': mcs_region_labels
    })
    return df_mcs


def detect_mcs_in_file(file_path, time_index=0):
    """
    Detect MCSs in a single file and return labeled regions and other necessary data.
    """
    # Load data
    ds, lat, lon, prec = load_data(file_path, time_index)
    
    # Apply smoothing
    prec_smooth = apply_smoothing(prec, method='gaussian', sigma=1)
    prec_to_use = prec_smooth

    # Identify moderate precipitation points
    moderate_prec_mask = identify_moderate_precipitation(prec_to_use, moderate_prec_threshold=2)

    # Cluster moderate precipitation points using DBSCAN
    moderate_labeled_regions = cluster_moderate_precipitation(moderate_prec_mask, lat, lon, eps_km=10, min_samples=5)

    # Calculate cluster areas
    cluster_areas = calculate_cluster_areas(moderate_labeled_regions, grid_spacing_km=4)

    # Filter clusters by area
    filtered_labeled_regions = filter_clusters_by_area(moderate_labeled_regions, cluster_areas, min_area_km2=500)

    # Calculate heavy precipitation percentage in each cluster
    heavy_prec_percentages = calculate_heavy_precip_percentage(filtered_labeled_regions, prec_to_use, heavy_prec_threshold=15)

    # Select clusters based on heavy precipitation percentage
    final_labeled_regions = select_clusters_by_heavy_precip_percentage(filtered_labeled_regions, heavy_prec_percentages, min_heavy_prec_percentage=20)

    # Return necessary data for tracking
    return {
        'file_path': file_path,
        'final_labeled_regions': final_labeled_regions,
        'lat': lat,
        'lon': lon,
        'prec': prec_to_use,
        'time': ds['time'].values
    }

def detect_mcs_in_file_new(file_path, time_index=0):
    """
    Detect MCSs in a single file and return labeled regions and other necessary data.
    """
    # Load data
    ds, lat, lon, precipitation = load_data(file_path, time_index)

    # Apply smoothing (optional)
    precipitation_smooth = apply_smoothing(precipitation, method='gaussian', sigma=1)
    precipitation_to_use = precipitation_smooth

    # Identify moderate precipitation points
    moderate_prec_mask = identify_moderate_precipitation(precipitation_to_use, moderate_prec_threshold=2)

    # Cluster moderate precipitation points using connected-component labeling
    labeled_regions, num_features = cluster_precipitation_regions(moderate_prec_mask)

    # Filter clusters by minimum size
    filtered_labeled_regions = filter_clusters_by_size(
        labeled_regions,
        min_area_km2=10000,  # Adjust as needed
        grid_spacing_km=4    # Adjust based on your data
    )

    # Identify clusters containing heavy precipitation
    final_labeled_regions = filter_clusters_with_heavy_precipitation(
        filtered_labeled_regions,
        precipitation_to_use,
        heavy_prec_threshold=15  # Adjust as needed
    )

    # Create MCS DataFrame
    df_mcs = create_mcs_dataframe(final_labeled_regions, lat, lon, precipitation_to_use)

    # Return necessary data for tracking
    return {
        'file_path': file_path,
        'final_labeled_regions': final_labeled_regions,
        'lat': lat,
        'lon': lon,
        'prec': precipitation_to_use,
        'time': ds['time'].values,
        'df_mcs': df_mcs
    }