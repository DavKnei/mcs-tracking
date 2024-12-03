# detection.py

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, label, binary_dilation
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def smooth_precipitation_field(precipitation, sigma=1):
    return gaussian_filter(precipitation, sigma=sigma)

def adjusted_connected_components(binary_mask, dilation_iterations=1):
    """
    Perform connected-component labeling after dilating the binary mask.
    """
    structure = np.ones((3, 3))
    dilated_mask = binary_dilation(binary_mask, structure=structure, iterations=dilation_iterations)
    clusters, num_clusters = label(dilated_mask, structure=structure)
    return clusters, num_clusters

def identify_convective_plumes(precipitation, clusters, heavy_threshold):
    """
    Identify convective plumes within clusters using watershed segmentation.
    """
    convective_plume_labels = np.zeros_like(clusters)
    cluster_labels = np.unique(clusters)
    cluster_labels = cluster_labels[cluster_labels != 0]  # Exclude background

    for label_value in cluster_labels:
        cluster_mask = clusters == label_value
        # Apply heavy precipitation threshold within the cluster
        heavy_mask = np.logical_and(precipitation >= heavy_threshold, cluster_mask)
        if np.any(heavy_mask):
            # Compute the distance transform
            distance = distance_transform_edt(heavy_mask)
            # Find local maxima
            coordinates = peak_local_max(
                distance,
                labels=cluster_mask,
                min_distance=3
            )
            if len(coordinates) == 0:
                continue  # Skip if no peaks are found
            # Create markers array
            markers = np.zeros_like(distance, dtype=int)
            for idx, (row, col) in enumerate(coordinates, start=1):
                markers[row, col] = idx
            # Apply watershed
            labels_ws = watershed(-distance, markers=markers, mask=cluster_mask)
            # Assign unique labels to convective plumes
            labels_ws += convective_plume_labels.max()
            convective_plume_labels += labels_ws

    return convective_plume_labels

def filter_mcs_candidates(clusters, convective_plumes, min_area_km2, min_plumes, grid_cell_area_km2):
    """
    Filter clusters to identify MCS candidates based on area and convective plumes.
    """
    mcs_candidate_labels = []
    cluster_labels = np.unique(clusters)
    cluster_labels = cluster_labels[cluster_labels != 0]

    for label_value in cluster_labels:
        cluster_mask = clusters == label_value
        area_km2 = np.sum(cluster_mask) * grid_cell_area_km2
        plumes_in_cluster = np.unique(convective_plumes[cluster_mask])
        num_plumes = len(plumes_in_cluster[plumes_in_cluster != 0])

        if area_km2 >= min_area_km2 and num_plumes >= min_plumes:
            mcs_candidate_labels.append(label_value)

    return mcs_candidate_labels

def detect_mcs_in_file(file_path, time_index=0):
    """
    Detect MCSs in a single file using adjusted connected-component labeling.
    """
    # Load data
    ds = xr.open_dataset(file_path)
    lat = ds['lat'].values
    lon = ds['lon'].values
    precipitation = ds['pr'][time_index, :, :].values  # Adjust variable name if necessary

    # Step 1: Smooth the precipitation field
    precipitation_smooth = smooth_precipitation_field(precipitation, sigma=1)

    # Step 2: Create binary mask for moderate precipitation
    moderate_threshold = 2  # mm
    binary_mask = precipitation_smooth >= moderate_threshold

    # Step 3: Perform adjusted connected-component labeling
    dilation_iterations = 1  # Adjust as needed
    clusters, num_clusters = adjusted_connected_components(binary_mask, dilation_iterations)

    # Step 4: Identify convective plumes within clusters
    heavy_threshold = 15  # mm
    convective_plumes = identify_convective_plumes(precipitation_smooth, clusters, heavy_threshold)

    # Step 5: Filter clusters based on area and plume criteria
    min_area_km2 = 10000  # Adjust as needed
    min_plumes = 2     # Adjust as needed
    grid_size = 4  # km
    grid_cell_area_km2 = grid_size**2
    mcs_candidate_labels = filter_mcs_candidates(clusters, convective_plumes, min_area_km2, min_plumes, grid_cell_area_km2)

    # Create final labeled regions for MCS candidates
    final_labeled_regions = np.where(np.isin(clusters, mcs_candidate_labels), clusters, 0)

    # Prepare detection result
    detection_result = {
        'file_path': file_path,
        'final_labeled_regions': final_labeled_regions,
        'moderate_prec_clusters': clusters,
        'lat': lat,
        'lon': lon,
        'precipitation': precipitation_smooth,
        'time': ds['time'].values[time_index],
        'convective_plumes': convective_plumes
    }

    return detection_result

