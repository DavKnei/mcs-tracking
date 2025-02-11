import numpy as np


def filter_mcs_candidates(
    clusters, convective_plumes, min_area_km2, min_nr_plumes, grid_cell_area_km2
):
    """
    Filter clusters to identify MCS candidates based on area and number of convective plumes.

    Parameters:
    - clusters: 2D array of cluster labels.
    - convective_plumes: 2D array of convective plume labels.
    - min_area_km2: Minimum area threshold for MCS candidate (in km²).
    - min_nr_plumes: Minimum number of convective plumes required for MCS candidate.
    - grid_cell_area_km2: Area of a single grid cell (in km²).

    Returns:
    - mcs_candidate_labels: List of cluster labels that meet the MCS criteria.
    """
    mcs_candidate_labels = []
    cluster_labels = np.unique(clusters)
    cluster_labels = cluster_labels[cluster_labels != 0]

    for label_value in cluster_labels:
        cluster_mask = clusters == label_value
        area_km2 = np.sum(cluster_mask) * grid_cell_area_km2
        plumes_in_cluster = np.unique(convective_plumes[cluster_mask])
        num_plumes = len(plumes_in_cluster[plumes_in_cluster != 0])

        if area_km2 >= min_area_km2 and num_plumes >= min_nr_plumes:
            mcs_candidate_labels.append(label_value)

    return mcs_candidate_labels


def compute_cluster_centers_of_mass(final_labeled_regions, lat, lon, precipitation):
    """Computes the precipitation-weighted center of mass for each labeled cluster.

    Args:
        final_labeled_regions (numpy.ndarray):
            2D integer array of cluster labels (excluding -1 or 0 for background).
        lat (numpy.ndarray):
            2D array of latitudes, same shape as final_labeled_regions.
        lon (numpy.ndarray):
            2D array of longitudes, same shape as final_labeled_regions.
        precipitation (numpy.ndarray):
            2D array of precipitation values, same shape as final_labeled_regions.

    Returns:
        dict:
            A dictionary mapping 'label_value' -> (center_lat, center_lon) in degrees.
            If a cluster has zero total precipitation, we store (np.nan, np.nan).
    """
    cluster_centers = {}
    unique_labels = np.unique(final_labeled_regions)
    # Exclude background label if it's 0 or -1
    unique_labels = unique_labels[unique_labels > 0]
    unique_labels = unique_labels.astype(str)  # Ensure keys are strings for JSON

    for label_value in unique_labels:
        mask = final_labeled_regions == int(label_value)
        # precipitation for this cluster only
        cluster_precip = np.where(mask, precipitation, 0.0)
        total_precip = np.sum(cluster_precip)

        if total_precip > 0:
            lat_weighted = np.sum(lat * cluster_precip)
            lon_weighted = np.sum(lon * cluster_precip)
            center_lat = lat_weighted / total_precip
            center_lon = lon_weighted / total_precip
            cluster_centers[label_value] = (center_lat, center_lon)
        else:
            # If zero total precipitation, we cannot define a weighted center
            cluster_centers[label_value] = (np.nan, np.nan)

    return cluster_centers
